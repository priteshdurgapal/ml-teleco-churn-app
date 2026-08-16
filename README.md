# Telco Customer Churn — Multi-Model Classification

## a. Problem Statement

Customer churn — customers discontinuing a service — is a major cost driver for subscription-based businesses like telecom providers. This project builds and compares multiple binary classification models to predict whether a customer will churn (`Churn`: Yes/No), based on their account, service, and billing information. The goal is to identify which model(s) best distinguish churners from non-churners, evaluate them with metrics appropriate for an imbalanced classification problem, and deploy an interactive tool for exploring the results.

## b. Dataset Description

- **Source:** [IBM Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Instances:** 7,043 customers
- **Features:** 19 raw predictor columns (customer demographics, account tenure, contract type, billing method, and subscribed services), expanding to ~30 features after one-hot encoding — well above the 12-feature minimum
- **Target:** `Churn` (Yes/No) — binary classification
- **Class balance:** ~27% churn / ~73% no-churn (imbalanced)
- **Preprocessing:** dropped the `customerID` identifier column; coerced `TotalCharges` to numeric (blank strings for zero-tenure customers filled with 0); one-hot encoded all categorical columns; scaled numeric features with `StandardScaler`; 80/20 stratified train-test split

## c. GitHub Repository Link

https://github.com/priteshdurgapal/ml-teleco-churn-app


## d. Models Used

All 5 models were trained on the same 80/20 stratified train-test split of the dataset above.


### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8070 | 0.8418 | 0.6584 | 0.5668 | 0.6092 | 0.4843 |
| Decision Tree | 0.7317 | 0.6530 | 0.4946 | 0.4866 | 0.4906 | 0.3085 |
| kNN | 0.7473 | 0.7716 | 0.5253 | 0.5000 | 0.5123 | 0.3422 |
| Naive Bayes | 0.6558 | 0.8092 | 0.4269 | 0.8663 | 0.5719 | 0.3951 |
| Random Forest (Ensemble) | 0.7942 | 0.8275 | 0.6419 | 0.5080 | 0.5672 | 0.4397 |


### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best all-around performer — highest AUC (0.842), F1 (0.609), and MCC (0.484). A linear model works well here because churn correlates fairly linearly with features like tenure, contract type, and monthly charges after encoding. |
| Decision Tree | Weakest model on every metric (AUC 0.653, MCC 0.309). A single unpruned tree overfits the training data and generalizes poorly — this is the expected motivation for using an ensemble instead. |
| kNN | Middling performance (AUC 0.772, MCC 0.342). Held back by the high-dimensional, sparse feature space created by one-hot encoding many categorical columns — distance metrics degrade in high dimensions ("curse of dimensionality"). |
| Naive Bayes | Lowest accuracy (0.656) but by far the highest recall (0.866) — catches most actual churners at the cost of many false alarms (precision only 0.427). Whether this is "worse" depends on business priorities: missing a real churner is usually costlier than a false alarm, so this model's recall-heavy trade-off could be more business-valuable despite weaker balanced metrics. |
| Random Forest (Ensemble) | Solid, well-rounded (AUC 0.828, MCC 0.440) — clearly improves over the single Decision Tree on every metric, demonstrating the value of ensembling many decorrelated trees to reduce overfitting. |

### Overall Winner for your dataset?

**Logistic Regression** — it leads on AUC, F1, and MCC simultaneously, which is a stronger claim than winning on any single metric. Plain Accuracy is unreliable here since the dataset is imbalanced (~27% churn); AUC, F1, and MCC all hold up better under imbalance, and Logistic Regression tops all three.

Caveat: if the business priority is *catching as many churners as possible* even at the cost of false positives, **Naive Bayes** is the more defensible pick despite its lower overall scores — the "right" model depends on which mistake (missing a churner vs. a false alarm) is more costly in practice, not just which one wins on paper.

