# Teen Mental Health Prediction Using Machine Learning

## Project Overview

Mental health challenges among teenagers have become an increasing concern, particularly with the rise of social media usage, academic pressure, and lifestyle changes. Early identification of individuals at risk of depression can help provide timely support and intervention.

This project uses machine learning techniques to predict depression risk among teenagers based on demographic, behavioral, academic, and mental health-related factors.

The project demonstrates a complete machine learning workflow including:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Handling imbalanced datasets
- Model training and evaluation
- Cross-validation
- Model interpretation

---

## Problem Statement

The goal of this project is to build a machine learning model capable of predicting whether a teenager is at risk of depression using factors such as:

- Anxiety levels
- Stress levels
- Sleep duration
- Social media usage
- Academic performance
- Physical activity
- Social interaction

Since mental health screening applications prioritize identifying at-risk individuals, the project focuses heavily on recall and F1-score rather than accuracy alone.

---

## Dataset Information

The dataset contains **1,200 records** representing teenagers and includes the following variables:

| Feature | Description |
|----------|------------|
| age | Age of the teenager |
| gender | Gender |
| daily_social_media_hours | Daily social media usage |
| sleep_hours | Average sleep duration |
| screen_time_before_sleep | Screen exposure before sleeping |
| physical_activity | Physical activity level |
| academic_performance | Academic performance score |
| social_interaction_level | Level of social interaction |
| stress_level | Stress score |
| anxiety_level | Anxiety score |
| addiction_level | Social media addiction level |
| platform_usage | Preferred social media platform |
| depression_label | Target variable |

### Target Variable

| Value | Meaning |
|---------|----------|
| 0 | Not Depressed |
| 1 | At Risk of Depression |

---

## Exploratory Data Analysis

### Dataset Shape

- Rows: 1200
- Columns: 14

### Missing Values

The dataset contained:

```text
0 missing values
```

No imputation or row removal was required.

### Class Distribution

The dataset was highly imbalanced:

| Class | Count |
|---------|---------:|
| Not Depressed (0) | 1169 |
| Depressed (1) | 31 |

This imbalance made accuracy an unreliable evaluation metric.

---

## Data Preprocessing

### Categorical Encoding

#### Gender

Binary encoding:

```text
Male   → 1
Female → 0
```

#### Social Interaction Level

Ordinal encoding:

```text
Low    → 0
Medium → 1
High   → 2
```

#### Platform Usage

One-Hot Encoding:

```text
Instagram
TikTok
Both
```

Converted into:

```text
platform_usage_Instagram
platform_usage_TikTok
platform_usage_Both
```

---

## Train-Test Split

The dataset was split into:

```text
80% Training Data
20% Testing Data
```

A stratified split was used to preserve class proportions across both sets.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

---

## Feature Scaling

Standardization was applied using StandardScaler:

```python
StandardScaler()
```

Scaling was performed only on the training set and then applied to the testing set to prevent data leakage.

---

## Models Used

### 1. Logistic Regression

```python
LogisticRegression(
    class_weight='balanced'
)
```

Class weighting was applied to address the severe class imbalance.

### 2. Random Forest

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
```

---

## Model Evaluation

### Logistic Regression Results

#### Accuracy

```text
97.1%
```

#### Classification Report

| Metric | Class 1 |
|----------|----------:|
| Precision | 0.46 |
| Recall | 1.00 |
| F1-score | 0.63 |

#### Confusion Matrix

```text
[[227   7]
 [  0   6]]
```

Interpretation:

- Correctly identified all 6 depression cases
- Missed 0 depression cases
- Generated 7 false alarms

---

### Random Forest Results

| Metric | Class 1 |
|----------|----------:|
| Precision | 1.00 |
| Recall | 0.17 |
| F1-score | 0.29 |

Although Random Forest achieved slightly higher overall accuracy, it failed to identify most depression cases.

---

## Model Comparison

| Model | Accuracy | Recall (Class 1) | F1 Score |
|---------|---------:|---------:|---------:|
| Logistic Regression | 97.1% | 1.00 | 0.63 |
| Random Forest | 98.0% | 0.17 | 0.29 |

### Selected Model

**Logistic Regression**

Reason:

The primary objective was identifying at-risk teenagers.

Logistic Regression achieved:

```text
100% Recall
```

meaning it successfully detected every depression case in the testing dataset.

---

## Feature Importance Analysis

The Logistic Regression model identified the following strongest predictors:

### Positive Predictors of Depression Risk

| Feature | Coefficient |
|----------|----------:|
| anxiety_level | 2.98 |
| stress_level | 2.43 |
| daily_social_media_hours | 2.09 |
| academic_performance | 0.59 |
| addiction_level | 0.55 |

### Protective Factors

| Feature | Coefficient |
|----------|----------:|
| sleep_hours | -3.39 |
| gender | -0.85 |
| platform_usage_Instagram | -0.35 |

### Key Findings

- Anxiety level was the strongest predictor of depression.
- Stress level showed a strong positive relationship with depression risk.
- Higher social media usage was associated with increased depression risk.
- Sleep duration was the strongest protective factor.

---

## Cross Validation

To ensure that results were not dependent on a single train-test split, 5-fold cross-validation was performed.

### Results

```text
F1 Scores:
[0.50, 0.43, 0.42, 0.44, 0.43]
```

### Performance Summary

| Metric | Value |
|----------|----------:|
| Mean F1 Score | 0.447 |
| Standard Deviation | 0.028 |

### Interpretation

The relatively low standard deviation indicates that model performance was stable across multiple folds.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

---

## Project Structure

```text
Teen-Mental-Health-Prediction/
│
├── data/
│   └── Teen_Mental_Health_Dataset.csv
│
├── notebooks/
│   └── mental_health_prediction.ipynb
│
├── images/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

---

## Future Improvements

Potential enhancements include:

- SMOTE oversampling for minority class balancing
- Hyperparameter tuning using GridSearchCV
- XGBoost implementation
- Deployment using Flask or Streamlit
- Real-time prediction interface

---

## Conclusion

This project demonstrates a complete end-to-end machine learning workflow for predicting depression risk among teenagers.

Despite the dataset's severe class imbalance, Logistic Regression achieved perfect recall on the minority class and successfully identified all depression cases in the testing dataset.

The analysis revealed that anxiety levels, stress levels, and social media usage were the strongest predictors of depression risk, while sleep duration acted as the strongest protective factor.

The project highlights the importance of evaluating models using recall and F1-score rather than relying solely on accuracy when working with imbalanced datasets.
