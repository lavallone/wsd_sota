from typing import Tuple, Any, Dict, Optional, List

import hydra
import torch
import os
import ntpath
from omegaconf import omegaconf, DictConfig

from src.consec_dataset import ConsecSample
from src.dependency_finder import EmptyDependencyFinder
from src.pl_modules import ConsecPLModule
from src.scripts.model.continuous_predict import Predictor
from src.sense_inventories import SenseInventory, WordNetSenseInventory
from src.utils.commons import execute_bash_command
from src.utils.hydra import fix
from src.utils.wsd import expand_raganato_path



def framework_evaluate(framework_folder: str, gold_file_path: str, pred_file_path: str) -> Tuple[float, float, float]:
    scorer_folder = f"{framework_folder}/Evaluation_Datasets"
    command_output = execute_bash_command(
        f"[ ! -e {scorer_folder}/Scorer.class ] && javac -d {scorer_folder} {scorer_folder}/Scorer.java; java -cp {scorer_folder} Scorer {gold_file_path} {pred_file_path}"
    )
    command_output = command_output.split("\n")
    p, r, f1 = [float(command_output[i].split("=")[-1].strip()[:-1]) for i in range(3)]
    return p, r, f1


def sample_prediction2sense(sample: ConsecSample, prediction: int, sense_inventory: SenseInventory) -> str:
    sample_senses = sense_inventory.get_possible_senses(
        sample.disambiguation_instance.lemma, sample.disambiguation_instance.pos
    )
    sample_definitions = [sense_inventory.get_definition(s) for s in sample_senses]

    for s, d in zip(sample_senses, sample_definitions):
        if d == sample.candidate_definitions[prediction].text:
            return s

    raise ValueError

def pos_eval(gold_file_path, pred_file_path):
    with open(gold_file_path, 'r') as gold_file: gold_lines = gold_file.readlines()
    with open(pred_file_path, 'r') as pred_file: pred_lines = pred_file.readlines()

    gold_dict = {line.split()[0]: line.split()[1:] for line in gold_lines }
    pred_dict = {line.split()[0]: line.split()[1] for line in pred_lines}

    corrects = {}
    wrongs = {}
    for key in gold_dict.keys():
        gold_labels = gold_dict[key]
        pred_label = pred_dict[key]
        if pred_label in gold_labels:
            corrects[key] =  pred_label
        else:
            wrongs[key] =  pred_label
    all_ris = len(corrects.keys()) / (len(corrects.keys())+len(wrongs.keys()))
    
    def map_pos(tag):
        pos_number = tag.split('%')[1][0]
        if pos_number == '1': return 'NOUN'
        elif pos_number == '2': return 'VERB'
        elif pos_number in ['3', '5']: return 'ADJ'
        elif pos_number == '4': return 'ADV'

    from collections import Counter
    corrects_list = [map_pos(v) for _,v in corrects.items()]
    c1 = Counter(corrects_list)
    wrongs_list = [map_pos(v) for _,v in wrongs.items()]
    c2 = Counter(wrongs_list)

    ris = {}
    for pos in c1.keys():
        ris[pos] = c1[pos]/(c1[pos]+c2[pos])
    ris["ALL"] = all_ris
    return ris


def raganato_evaluate(
    raganato_path: str,
    wsd_framework_dir: str,
    module: ConsecPLModule,
    predictor: Predictor,
    wordnet_sense_inventory: WordNetSenseInventory,
    samples_generator: DictConfig,
    prediction_params: Dict[Any, Any],
    fine_grained_evals: Optional[List[str]] = None,
    reporting_folder: Optional[str] = None,
) -> Tuple[float, float, float, Optional[List[Tuple[str, float, float, float]]]]:

    # load tokenizer
    tokenizer = hydra.utils.instantiate(module.hparams.tokenizer.consec_tokenizer)

    # instantiate samples
    consec_samples = list(hydra.utils.instantiate(samples_generator, dependency_finder=EmptyDependencyFinder())())

    # predict
    path_dir = os.path.dirname(expand_raganato_path(raganato_path)[1])
    preds_file = os.path.join( "/".join(os.getcwd().split("/")[:-3]), "predictions/{}_predictions.txt".format(ntpath.basename(path_dir)) )
    if not os.path.exists(preds_file):
        print("\nCREATING CONSEC PREDICTION FILE...\n")
        disambiguated_samples = predictor.predict(
            consec_samples,
            already_kwown_predictions=None,
            reporting_folder=reporting_folder,
            **dict(module=module, tokenizer=tokenizer, **prediction_params),
        )
        with open(preds_file, "w") as f:
            for sample, idx in disambiguated_samples:
                f.write(f"{sample.sample_id} {sample_prediction2sense(sample, idx, wordnet_sense_inventory)}\n")

    # here I need to compute POS evaluation --> [NOUN, VERB, ADJ, ADV]
    # starting from the gold and prediction file
    pos_dict = pos_eval(expand_raganato_path(raganato_path)[1], preds_file)

    # compute metrics
    p, r, f1 = framework_evaluate(
        wsd_framework_dir,
        gold_file_path=expand_raganato_path(raganato_path)[1],
        pred_file_path=preds_file,
    )

    # fine grained eval
    fge_scores = None

    if fine_grained_evals is not None:
        fge_scores = []
        for fge in fine_grained_evals:
            _p, _r, _f1 = framework_evaluate(
                wsd_framework_dir,
                gold_file_path=expand_raganato_path(fge)[1],
                pred_file_path=preds_file,
            )
            fge_scores.append((fge, _p, _r, _f1))

    return p, r, f1, fge_scores, pos_dict


@hydra.main(config_path="../../../conf/test", config_name="raganato")
def main(conf: omegaconf.DictConfig) -> None:

    fix(conf)

    # load module
    # todo decouple ConsecPLModule
    module = ConsecPLModule.load_from_checkpoint(conf.model.model_checkpoint)
    module.to(torch.device(conf.model.device if conf.model.device != -1 else "cpu"))
    module.eval()
    module.freeze()
    module.sense_extractor.evaluation_mode = True  # no loss will be computed even if labels are passed

    # instantiate sense inventory
    sense_inventory = hydra.utils.instantiate(conf.sense_inventory)

    # instantiate predictor
    predictor = hydra.utils.instantiate(conf.predictor)

    # evaluate
    p, r, f1, fge_scores, pos_dict = raganato_evaluate(
        raganato_path=conf.test_raganato_path,
        wsd_framework_dir=conf.wsd_framework_dir,
        module=module,
        predictor=predictor,
        wordnet_sense_inventory=sense_inventory,
        samples_generator=conf.samples_generator,
        prediction_params=conf.model.prediction_params,
        fine_grained_evals=conf.fine_grained_evals,
        reporting_folder=".",  # hydra will handle it
    )
    # print(f"# p: {p}")
    # print(f"# r: {r}")
    # print(f"# f1: {f1}")

    if fge_scores:
        for fge, p, r, f1 in fge_scores:
            print(f'# {fge}: ({p:.1f}, {r:.1f}, {f1:.1f})')
    
    # POS evaluation
    print()
    for k,v in pos_dict.items():
        print(f"{k} | {v*100:.1f}")
    print()

if __name__ == "__main__":
    main()
