# Binary Prediction of Smoker Status

## Project Overview

![](biosignal.jpg)
This project focuses on developing a binary classification model to predict a patient’s smoking status based on various health indicators or bio-signals. The goal is to utilize these features to accurately classify whether a patient is a smoker or not.

## Dataset Description

The dataset used in this project was generated from a deep learning model trained on the "Smoker Status Prediction using Bio-Signals" dataset. While the feature distributions in this dataset are close to, but not exactly the same as, the original dataset, it offers a valuable opportunity for model development and evaluation.

### Dataset Details

- **Train and Test Sets:** The dataset is divided into training and test sets. The training set is used to train the model, while the test set is used to evaluate its performance.
- **Features:** The dataset includes various health indicators or bio-signals that are used to predict smoking status.
- **Original Dataset:** You may also use the original dataset for further exploration and to assess whether incorporating it into the training process improves model performance.

## Objective

The primary objective of this project is to build a binary classification model that predicts a patient's smoking status. The model will be trained using the provided dataset and evaluated based on its accuracy and effectiveness in distinguishing between smokers and non-smokers.

## Citation

- **Authors:** Walter Reade, Ashley Chow
- **Year:** 2023
- **Title:** Binary Prediction of Smoker Status using Bio-Signals
- **Source:** Kaggle
- **Link:** [Kaggle Competition](https://kaggle.com/competitions/playground-series-s3e24)

## Methodology

1. **Data Preprocessing:** Clean and preprocess the dataset to handle missing values, normalize features, and split the data into training and test sets.
2. **Feature Engineering:** Extract relevant features from the bio-signals and health indicators that may be predictive of smoking status.
3. **Model Selection:** Experiment with various binary classification algorithms (e.g., Logistic Regression, Random Forest, Gradient Boosting) to find the most effective model.
4. **Evaluation:** Assess model performance using metrics such as accuracy, precision, recall, and F1-score. Use cross-validation to ensure robustness.
5. **Model Tuning:** Fine-tune model parameters to improve performance based on evaluation metrics.

## Installation

To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yash-raj202134/Binary-prediction-of-smoker-status.git
cd Binary-prediction-of-smoker-status
pip install -r requirements.txt
```
Now execute:
```bash
python app.py
```
