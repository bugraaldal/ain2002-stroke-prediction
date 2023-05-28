# Stroke Prediciton with Gradient Boosting on Synthetical and Real-World Data

This repository is the official implementation of "Stroke Prediciton with Gradient Boosting".

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. This study is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. This study uses [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and a synthetically created dataset drived from it for a Kaggle competition, [Binary Classification with a Tabular Stroke Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s3e2/data).

The author of the real world data we use states that the source of this dataset is confidential therefore we do not know how it's collection or sampling were made. The synthetic data's description states that it was created by a deep learning model tranied on the real world data that we currently use. Thus, if the real world data has any bias in its collection, that means our synthetic data has the bias as well. However, we do not have any information about the real world data's collection or sampling. We will mention some attributes that makes us raise some eyebrows later on in this study.

## Requirements

To install requirements:    
`cd` into `ain2002-stroke-predictions` folder and run the following command in the terminal:
```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python modeling.py
```
`modeling.ipynb` file includes experimental submissions and `modeling.py` only outputs the final submission CSV file.

## Evaluation
AUROC was used for the model's evaluation score.

![image](https://github.com/bugraaldal/ain2002-stroke-prediction/assets/61989756/02c8a210-b899-448a-a670-ff2c26b8a6e6)

## Results
```
__________________________________________________
[0]	validation_0-logloss:0.68453
[930]	validation_0-logloss:0.12712
Val AUC: 0.8864452229574844
__________________________________________________
[0]	validation_0-logloss:0.68455
[645]	validation_0-logloss:0.12764
Val AUC: 0.8772976947363869
__________________________________________________
[0]	validation_0-logloss:0.68448
[583]	validation_0-logloss:0.12509
Val AUC: 0.8803521216768916
__________________________________________________
[0]	validation_0-logloss:0.68457
[775]	validation_0-logloss:0.12507
Val AUC: 0.8954925017041582
__________________________________________________
[0]	validation_0-logloss:0.68453
[756]	validation_0-logloss:0.12277
Val AUC: 0.8922322848703216
__________________________________________________
[0]	validation_0-logloss:0.68454
[650]	validation_0-logloss:0.12744
Val AUC: 0.8863569967864446
__________________________________________________
[0]	validation_0-logloss:0.68453
[704]	validation_0-logloss:0.12507
Val AUC: 0.8835546033910042
__________________________________________________
[0]	validation_0-logloss:0.68451
[775]	validation_0-logloss:0.13057
Val AUC: 0.8740654180326981
__________________________________________________
[0]	validation_0-logloss:0.68457
[696]	validation_0-logloss:0.12831
Val AUC: 0.8828350699516343
__________________________________________________
[0]	validation_0-logloss:0.68456
[772]	validation_0-logloss:0.12519
Val AUC: 0.8843336471148332

Mean Val AUC: 0.8842965561221856
OOF AUC: 0.8840207775423436
```

![image](https://github.com/bugraaldal/ain2002-stroke-prediction/assets/61989756/3bc81b4c-eca1-4e35-824f-40778328e5ca)


## Contributions
Buğra Aldal: Github   
Hamdi Alakkad: Report   
MHD Ghaith Saeedy: Report    
