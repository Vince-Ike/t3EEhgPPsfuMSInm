# Customer Happiness Prediction Using XGBoost


**Developing a Machine Learning Classification Model to Predict Customer Happiness and Support Data-Driven Customer Experience Decisions**

---

## 1. Project Overview

Customer satisfaction is an important indicator of business performance, customer retention, and long-term growth. Understanding the factors associated with positive customer experiences can help organizations identify opportunities to improve their services and strengthen customer relationships.

This project focuses on developing a supervised machine learning classification model to predict customer happiness using the **ACME Happiness Survey 2020** dataset.

Using Python and XGBoost (Extreme Gradient Boosting), I developed a binary classification model to predict whether a customer reported a positive happiness outcome based on six survey responses.

The project emphasizes model development, prediction probability analysis, classification threshold optimization, and performance evaluation. Rather than relying exclusively on the default classification threshold of 0.50, I investigated alternative thresholds to better understand the trade-offs between accuracy, precision, recall, and F1 score.

The overall objective was to develop a classification approach that could provide meaningful predictive insights while demonstrating a structured machine learning workflow.

## 2. Business Problem

Organizations collect customer feedback to understand satisfaction levels and identify opportunities for service improvement. However, identifying patterns associated with positive customer experiences can be challenging.

The business problem addressed in this project is:

**Can machine learning identify patterns in customer survey responses that help predict whether a customer is likely to report a positive experience?**

By developing a predictive classification model, organizations could potentially use customer survey data to:

- Identify patterns associated with positive customer experiences.
- Better understand the relationship between survey responses and customer happiness.
- Support customer experience analysis and service improvement initiatives.
- Provide data-driven insights for business decision-making.

The model is intended to support customer experience analysis rather than replace direct customer feedback or establish causation.

## 3. Project Objectives

The primary objectives were to:

1. Prepare the survey dataset for supervised machine learning.
2. Develop an XGBoost binary classification model.
3. Configure model parameters to control complexity and improve generalization.
4. Generate prediction probabilities to evaluate classification confidence.
5. Investigate different classification thresholds to understand their impact on model performance.
6. Evaluate the model using multiple classification metrics.
7. Examine feature importance to explore the contribution of survey variables to model predictions.

## 4. Dataset Description

**Dataset:** ACME Happiness Survey 2020

The dataset contains customer survey responses collected to evaluate customer satisfaction with an ordering and delivery experience.

It consists of six independent predictor variables (X1–X6) and one dependent target variable (Y). Each predictor represents a specific aspect of the customer's experience, including delivery timeliness, order accuracy, pricing satisfaction, courier satisfaction, and application usability.

### Target Variable

| Variable | Description | Values |
|---|---|---|
| Y | Customer happiness classification | 0 = Unhappy, 1 = Happy |

### Feature Descriptions

The following six features were used as independent variables in the classification model.

| Feature | Description |
|---|---|
| X1 | My order was delivered on time. |
| X2 | Contents of my order were as I expected. |
| X3 | I ordered everything I wanted to order. |
| X4 | I paid a good price for my order. |
| X5 | I am satisfied with my courier. |
| X6 | The app makes ordering easy for me. |

### Feature Interpretation

Each feature captures a different dimension of the customer experience:

- **Delivery Performance (X1):** Evaluates whether the customer's order arrived on time.
- **Order Accuracy (X2):** Measures whether the delivered order matched the customer's expectations.
- **Order Completeness (X3):** Evaluates whether customers were able to order everything they intended to purchase.
- **Price Satisfaction (X4):** Captures customers' perceptions of the value they received for the price paid.
- **Courier Satisfaction (X5):** Reflects customer satisfaction with the courier responsible for delivery.
- **Application Usability (X6):** Measures how easily customers can navigate and place orders through the application.

Together, these features provide a structured representation of the ordering and delivery experience, allowing the machine learning model to identify patterns associated with customer happiness.

The target variable (Y) serves as the classification label that the model attempts to predict.

## 5. Technologies and Tools

The project was implemented using Python and the following libraries:

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| Pandas | Dataset loading and data manipulation |
| NumPy | Numerical operations and threshold analysis |
| Scikit-learn | Data splitting and model evaluation |
| XGBoost | Gradient boosting classification |
| GitHub | Version control and project documentation |

### Machine Learning Algorithm: XGBoost

XGBoost is an optimized gradient boosting algorithm that builds an ensemble of decision trees sequentially. Each subsequent tree attempts to correct errors made by the preceding trees.

I selected XGBoost because it is well suited to structured tabular datasets and can capture nonlinear relationships and interactions between predictor variables.

Its regularization capabilities also provide mechanisms for controlling model complexity and reducing overfitting.

## 6. Project Workflow

The project followed a structured machine learning development process.

### Step 1: Data Loading and Preparation

The dataset was loaded into a Pandas DataFrame.

The predictor variables were separated from the target variable:

- `X`: Independent variables containing customer survey responses.
- `y`: Dependent variable containing the happiness outcome (`Y`).

This separation established the input features and prediction target for supervised learning.

### Step 2: Training and Testing Data Split

The dataset was divided into training and testing subsets using Scikit-learn's `train_test_split()`.

The following configuration was used:

| Parameter | Configuration |
|---|---|
| Test size | 20% |
| Training size | 80% |
| Random state | 21 |
| Stratification | Enabled |

Stratified sampling was used to preserve the original target-class distribution across the resulting subsets.

This approach helps ensure that the model is trained and evaluated using representative class distributions.

### Step 3: XGBoost Model Development

An XGBoost classifier was developed to predict customer happiness based on the six survey features.

The model was configured using the following hyperparameters:

```python
xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.03,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=3,
    scale_pos_weight=0.85,
    random_state=21,
    eval_metric="logloss",
    n_jobs=1
)
```

### Hyperparameter Configuration and Rationale

| Hyperparameter | Value | Purpose |
|---|---:|---|
| `n_estimators` | 50 | Number of boosting trees constructed during training. |
| `max_depth` | 2 | Restricts individual tree depth to control model complexity. |
| `learning_rate` | 0.03 | Controls how much each successive tree contributes to the model. |
| `min_child_weight` | 3 | Controls the minimum sum of instance weights required in a child node. |
| `subsample` | 0.8 | Uses 80% of the training observations for each boosting iteration. |
| `colsample_bytree` | 0.7 | Uses 70% of available features when constructing each tree. |
| `reg_lambda` | 3 | Applies L2 regularization to discourage overly complex models. |
| `scale_pos_weight` | 0.85 | Adjusts the relative weighting of the positive class during training. |
| `random_state` | 21 | Ensures reproducibility of randomized operations. |
| `eval_metric` | `logloss` | Measures the difference between predicted probabilities and actual class labels. |
| `n_jobs` | 1 | Restricts execution to one CPU thread. |

### Model Configuration Rationale

The selected hyperparameters emphasize controlling model complexity while allowing the classifier to identify meaningful patterns in the survey responses.

In particular:

- **Shallow decision trees (`max_depth=2`):** Limit the complexity of individual trees and reduce the risk of overfitting.
- **Conservative learning rate (`learning_rate=0.03`):** Allows the model to learn gradually through smaller boosting updates.
- **Limited boosting iterations (`n_estimators=50`):** Restricts the number of trees used to construct the ensemble.
- **Regularization (`reg_lambda=3`):** Adds a penalty to discourage excessive model complexity.
- **Row and feature subsampling:** Introduce additional variation during training and can help improve generalization.
- **Class weighting (`scale_pos_weight=0.85`):** Slightly reduces the relative weight assigned to positive-class observations during training compared with negative-class observations.

The combination of these parameters establishes a relatively conservative XGBoost configuration designed to control model complexity while learning relationships between customer survey responses and happiness outcomes.

The model was subsequently trained using the designated training dataset.

### Step 4: Prediction Probability Analysis

Rather than immediately using the model's default class predictions, I used the XGBoost `predict_proba()` method.

This method generates probability estimates for each class.

The probability associated with class 1 was extracted:

```python
val_probs = xgb_model.predict_proba(X_val)[:, 1]
```

These probabilities represent the model's estimated likelihood that an observation belongs to the positive happiness class.

This step was important because it allowed me to investigate how different classification thresholds influence the final predictions.

### Step 5: Classification Threshold Optimization

By default, binary classification commonly uses a threshold of 0.50.

However, this threshold is not necessarily the most appropriate choice for every classification problem.

I evaluated alternative thresholds ranging from 0.30 to 0.70, using increments of 0.01.

For each threshold, the predicted class was determined using:

```python
val_pred = (val_probs >= threshold).astype(int)
```

The model was evaluated using four classification metrics:

**Accuracy**

Measures the proportion of all predictions that were correct.

**Precision**

Measures the proportion of positive predictions that were actually positive.

**Recall**

Measures the proportion of actual positive observations correctly identified by the model.

**F1 Score**

Calculates the harmonic mean of precision and recall, providing a balanced measure of the two metrics.

The threshold evaluation results were stored in a Pandas DataFrame and sorted by accuracy, followed by F1 score.

This allowed me to examine the trade-offs between different classification thresholds and identify a candidate threshold based on the selected evaluation criteria.

### Step 6: Final Model Evaluation

After selecting the candidate classification threshold, the model was fitted again using the designated training subset.

The model generated positive-class probabilities for the test observations.

The selected threshold was then applied to convert those probabilities into binary predictions.

Final model performance was examined using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

The classification report provides additional insight into class-specific precision, recall, and F1 score.

The confusion matrix provides a breakdown of true positives, true negatives, false positives, and false negatives.

These metrics help provide a more complete understanding of classification performance than accuracy alone.

### Step 7: Feature Importance Analysis

The final stage involved examining the feature importance values generated by XGBoost.

Feature importance provides an indication of how predictor variables contribute to the model's decision-making process.

This analysis can help identify survey variables that the model relies on when predicting customer happiness.

Feature importance should be interpreted as a measure of predictive contribution rather than evidence that a particular survey response causes customer happiness.

## 7. Model Evaluation and Performance Considerations

An important part of this project was recognizing that model performance cannot be evaluated using accuracy alone.

Different classification metrics provide different perspectives on predictive performance.

For example:

- Higher accuracy indicates that a greater proportion of predictions are correct.
- Higher precision reduces the number of incorrectly predicted positive outcomes.
- Higher recall reduces the number of actual positive outcomes that the model misses.
- A higher F1 score indicates a stronger balance between precision and recall.

Threshold optimization provided an opportunity to investigate these trade-offs and understand how classification decisions influence the final evaluation metrics.

### Evaluation Integrity Consideration

To maintain separation between model development and final performance evaluation, the dataset was divided into three subsets: training, validation, and testing.

The original dataset was initially divided into 80% training data and 20% testing data. The training subset was subsequently divided into 75% fitting data and 25% validation data.

The workflow consisted of the following stages:

- **Initial Training (60%):** Used to fit the initial XGBoost model.
- **Validation (20%):** Used to evaluate prediction probabilities and select the classification threshold.
- **Final Model Retraining (80%):** After threshold selection, the XGBoost model was retrained using the entire original training dataset, including the observations previously used for validation.
- **Final Testing (20%):** The retrained model was evaluated using the untouched test dataset and the selected classification threshold.

This approach allows the final model to learn from the entire available training dataset while preserving an independent test dataset for performance evaluation.

One consideration is that retraining on additional observations can change predicted probabilities, meaning the validation-selected threshold may not remain optimal. Nevertheless, keeping the threshold fixed during final testing provides an independent assessment of the resulting classification approach.

## 8. Key Skills Demonstrated

This project demonstrates practical experience in the following areas:

### Machine Learning

- Supervised learning
- Binary classification
- Gradient boosting
- XGBoost model development
- Hyperparameter configuration
- Model regularization

### Data Science

- Structured dataset preparation
- Feature and target separation
- Stratified sampling
- Prediction probability analysis
- Classification threshold evaluation
- Feature importance analysis

### Model Evaluation

- Accuracy
- Precision
- Recall
- F1 score
- Classification reports
- Confusion matrices
- Performance trade-off analysis

### Technical Tools

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Git and GitHub

## 9. Repository Structure

```text
t3EEhgPPsfuMSInm/
│
├── Project_HC.py
├── README.md
└── requirements.txt
```

The dataset is required to execute the project and may need to be downloaded separately.

## 10. How to Run the Project

### Prerequisites

- Python 3.9 or later
- pip
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Vince-Ike/t3EEhgPPsfuMSInm.git
```

### Step 2: Navigate to the Project Directory

```bash
cd t3EEhgPPsfuMSInm
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**macOS/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install numpy pandas scikit-learn xgboost
```

### Step 5: Configure the Dataset Path

Ensure that `ACME-HappinessSurvey2020.csv` is available locally.

Update the file path in `Project_HC.py` to reference the dataset location on your computer.

For example:

```python
df = pd.read_csv("ACME-HappinessSurvey2020.csv")
```

### Step 6: Execute the Project

```bash
python Project_HC.py
```

The script will train the XGBoost classifier, generate prediction probabilities, evaluate classification thresholds, display model performance metrics, and output feature importance values.

## 11. Conclusion

This project provided an opportunity to apply supervised machine learning techniques to a practical customer experience prediction problem.

Through the development of an XGBoost classification model, I explored how structured survey responses can be used to identify patterns associated with customer happiness.

One of the most valuable aspects of this project was investigating prediction probabilities and classification threshold optimization. This demonstrated the importance of looking beyond default model predictions and evaluating how different classification decisions affect accuracy, precision, recall, and F1 score.

I also gained practical experience configuring gradient boosting parameters, evaluating model performance, and examining feature importance to better understand model behavior.

**From a professional perspective, this project demonstrates my ability to approach a machine learning problem systematically, translate a business objective into a predictive modeling task, implement an appropriate classification algorithm, and critically evaluate its performance.**

My contribution reflects an understanding that successful data science is not simply about building a model. It also involves understanding the business problem, selecting appropriate techniques, evaluating results responsibly, and communicating meaningful insights to support informed decision-making.

---

## Author

**Vince Ike Osayande**

M.S. Computer Science – Artificial Intelligence and Machine Learning

AWS Certified Machine Learning Engineer – Associate

- [GitHub Profile](https://github.com/Vince-Ike)
- [Project Repository](https://github.com/Vince-Ike/t3EEhgPPsfuMSInm)
