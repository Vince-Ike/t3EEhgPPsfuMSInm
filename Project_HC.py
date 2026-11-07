# ============================================================
# XGBoost Classification Model
# ============================================================
# Purpose:
# Prepare the dataset
# train an XGBoost classifier, and evaluate model
# performance using classification metrics.
# ============================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("/Users/vinceike/Desktop/Apziva Projects/ACME-HappinessSurvey2020.csv")
df.shape
# =============================================================
# Split features and target columns
#==============================================================
X = df.drop(columns=['Y'])
y = df['Y']
# ============================================================
# Create Training and Testing Sets
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=21,
    stratify=y
)
#==============================================================
# Split data into Training and Validation
#===============================================================
X_fit, X_val, y_fit, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=21,
    stratify=y
)
#=================================================================
# Initialize XGBoost Model
#=================================================================
xgb_model= XGBClassifier(
    n_estimators =200,
    max_depth = 5,
    learning_rate = 0.05,
    min_child_weight = 8,
    subsample = 0.8,
    colsample_bytree= 0.8,
    reg_lambda = 5,
    random_state = 21,
    eval_metric ='logloss'

)

#=================================================================
# Train Model
#=================================================================
xgb_model.fit(X_fit, y_fit)

# ============================================================
# Generate Validation probabilities
# ============================================================
val_probs = xgb_model.predict_proba(X_val)[:,1]

print("\nValidation Probabilities:")
print(val_probs)

# ============================================================
#  Evaluate Different Thresholds
# ============================================================
threshold_results = []

thresholds = np.arange(0.30, 0.71, 0.01)

for threshold in thresholds:
    val_pred = (val_probs >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_val, val_pred),
        "Precison": precision_score(y_val, val_pred, zero_division=0),
        "Recall": recall_score(y_val, val_pred, zero_division=0),
        "F1 Score": f1_score(y_val, val_pred, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)
#Display the 10 best thresholds by accuracy

print("\nTop 10 Classification Thresholds")
print("=" * 40)

print(
    threshold_df.sort_values(
        by=["Accuracy","F1 Score"],
        ascending=False
    ).head(10).to_string(index=False)
)
# ============================================================
# Get the Best threshold
# ============================================================

best_threshold = threshold_df.sort_values(
        by=["Accuracy","F1 Score"],
        ascending=False
    ).iloc[0]["Threshold"]

print(f"\nBest Classification Threshold: {best_threshold:.2f}")
# ============================================================
# Retrain Model on Entire Training Set 
# ===========================================================
xgb_model.fit(X_train, y_train)
y_pred_probs = xgb_model.predict_proba(X_test)[:,1]
y_pred = (y_pred_probs >= best_threshold).astype(int)


# ============================================================
# Evaluate Final Model
# ============================================================
print("\nFinal Test Model Performance")
print("=" * 45)

print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# Display Prediction Probabilities
# ============================================================
prediction_results = pd.DataFrame({
    "Actual": y_test.values,
    "Probability": y_pred_probs,
    "Predicted": y_pred
})

print("\nPrediction Probabilities and Classifications")
print("=" * 55)
print(prediction_results.to_string(index=False))
# ============================================================
# Feature Importance
# =============================================================
print(xgb_model.feature_importances_)
# Results of feature importance is discuss in the readme file