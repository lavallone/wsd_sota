wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
unzip WSD_Evaluation_Framework.zip
rm WSD_Evaluation_Framework.zip
mv ALLamended WSD_Evaluation_Framework/Evaluation_Datasets
cp -r WSD_Evaluation_Framework consec/data
mv WSD_Evaluation_Framework esc/data
