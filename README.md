# Predicting Funding at Donorschoose.org
 
## Data Source

https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data

## Data Processing

dataprocessing.ipynb, dataprocessing.py (for easy import into other notebooks)

## EDA

eda.ipynb

## Gaussian Naive Bayes

bayes.ipynb (uses processed datasets from dataprocessing.py)

## XGBoost

xgboost.ipynb (uses processed datasets from dataprocessing.py)

## Random Forest

## Decision Tree

## NLP
We applied topic modeling using **Latent Dirichlet Allocation (LDA)** to extract underlying themes from the essay texts submitted by teachers. 

- `essays_text_preprocess.py`: Cleans and preprocesses essay text data (e.g., tokenization, stopword removal, lemmatization).
- `essays.py`: Implements LDA topic modeling and integrates topic distributions as features into the final dataset.
