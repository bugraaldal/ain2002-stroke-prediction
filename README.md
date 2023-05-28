# Stroke Prediciton with Gradient Boosting on Synthetical and Real-World Data

This repository is the official implementation of "Stroke Prediciton with Gradient Boosting".

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. This study is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. This study uses (Stroke Prediction Dataset)[https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset] and a synthetically created dataset drived from it for a Kaggle competition, (Binary Classification with a Tabular Stroke Prediction Dataset)[https://www.kaggle.com/competitions/playground-series-s3e2/data].

The author of the real world data we use states that the source of this dataset is confidential therefore we do not know how it's collection or sampling were made. The synthetic data's description states that it was created by a deep learning model tranied on the real world data that we currently use. Thus, if the real world data has any bias in its collection, that means our synthetic data has the bias as well. However, we do not have any information about the real world data's collection or sampling. We will mention some attributes that makes us raise some eyebrows later on in this study.

## Requirements

To install requirements:

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
![image](https://github.com/bugraaldal/ain2002-stroke-prediction/assets/61989756/3bc81b4c-eca1-4e35-824f-40778328e5ca)





## Contributing
what each of us has done

>📋  Pick a licence and describe how to contribute to your code repository. 
