# 🧠 Predicting Mental Health in Tech Workers

This project explores whether we can predict an individual's current mental health condition using workplace and demographic features. We apply various machine learning techniques to the [OSMI Mental Health in Tech Survey 2016](https://osmihelp.org/research) dataset and analyze feature importance and model performance.

---

## 📌 Motivation

A surge of anxiety posts by tech workers on platforms like **Blind** reflects the growing concern about mental health in the tech industry. We aim to answer:

- Are there strong predictors of current mental health conditions?
- Can we build a model to **help companies detect and support employees at risk**?

---

## 📊 Dataset

- Source: OSMI Mental Health in Tech Survey 2016  
- Respondents: 1,166 tech professionals  
- Target variable: `current_mh_disorder` (Yes / No / Maybe)

### Key Features:
- Mental health history  
- Family history  
- Perceived workplace support  
- Company size, remote status, benefits  
- Age, gender, and country

---

## 🧹 Data Processing

- **Filtering**: Removed non-tech participants
- **Missing values**:
  - Logical NAs → `"Not applicable"`
  - Random missing → Imputed via Random Forest
  - Dropped columns with >70% missing or open-ended responses
- **Feature Engineering**:
  - Categorical: One-Hot Encoding
  - Numerical: Standardization
  - Target: Label Encoding

---

## 🧠 Modeling Approach

We trained and compared the following models:

- Naive Bayes  
- Logistic Regression  
- SGD  
- SVM  
- Decision Tree  
- KNN  
- Neural Network (MLP)  
- Random Forest  
- XGBoost  
- ✅ **XGBoost RF (best)**

Used `GridSearchCV` to optimize hyperparameters for XGBoost RF.

---

## ✅ Best Model: XGBoost RF

| Class    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| Maybe    | 0.72      | 0.44   | 0.55     |
| No       | 0.88      | 0.75   | 0.81     |
| Yes      | 0.73      | 0.97   | 0.84     |

- Model favors **high recall** on "Yes" to avoid missing actual mental health cases
- Top 7 features used still achieve comparable accuracy, demonstrating model simplicity and robustness

---

## 🔍 Feature Importance (Top Variables)

| Feature                                             | Description                                            |
|-----------------------------------------------------|--------------------------------------------------------|
| `past_mh_disorder_Yes / No`                         | Self-reported history of mental illness               |
| `interfere_work_untreated_Not applicable to me`     | Belief they don’t have untreated issues               |
| `family_history_No`                                 | Absence of family history of mental illness           |
| `mh_diagnosed_prof_Yes`                             | Diagnosis by a mental health professional             |
| `interfere_work_treated_Not applicable to me`       | Perception of how treated issues affect work          |

📌 The model relies on **self-awareness**, **past experience**, and **family background** to assess current risk.

---

## 💬 Discussion

- We infer that XGBoost RF outperformed other models because each tree in the ensemble is trained independently, making it less prone to overfitting and better at generalizing when dealing with high-dimensional data.
- We chose recall as the primary evaluation metric to prioritize identifying individuals who truly have mental health issues. It is better to include a few false positives than to miss those who genuinely need support.
- The simplified model (recall for "Yes" = 0.95) achieved performance comparable to the full model, indicating that a subset of features is sufficient to address the prediction task.
- According to feature importance analysis, the most influential features in predicting mental health status include:
  - interfere_work_untreated_Not applicable to me: The respondent believes they do not have untreated mental health issues — this was the top-contributing feature.
  - past_mh_disorder_Yes / No: A history of mental health disorders strongly affects prediction outcomes.
  - family_history_No: Despite having lower average importance, it was used extensively by the model, suggesting it serves as a stable reference.

- Overall, the model heavily relies on self-awareness, personal history, and family background to assess mental health risk.
- In the future, we hope to explore modifiable and intervention-oriented factors, such as company culture, organizational policies, and peer support.
