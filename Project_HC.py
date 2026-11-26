
# ============================================================
# XGBoost Classification Model
# ============================================================
# Purpose:
# Prepare the dataset, train the XGBoost classifier, evaluate
# cross-validation performance, examine prediction
# probabilities, compare against a majority-class baseline,
# and evaluate final test performance.
# ============================================================
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_validate
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# Load Dataset
# ============================================================
# Get the directory containing TDM.py
BASE_DIR = Path(__file__).resolve().parent

# Move up one directory to Apziva Projects
DATA_PATH = BASE_DIR.parent / 'ACME-HappinessSurvey2020.csv'

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)

# ============================================================
# Split Features and Target
# ============================================================

X = df.drop(columns=["Y"])
y = df["Y"]

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Distribution Percentage:")
print((y.value_counts(normalize=True) * 100).round(2))

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

# ============================================================
# Split Training Data into Fitting and Validation Sets
# ============================================================

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=21,
    stratify=y_train
)

print("\nDataset Split Sizes:")
print(f"Model fitting data: {X_fit.shape}")
print(f"Validation data:    {X_val.shape}")
print(f"Training data:      {X_train.shape}")
print(f"Testing data:       {X_test.shape}")

# ============================================================
# Initialize XGBoost Model
# ============================================================

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

# ============================================================
# Train Model on Fitting Data
# ============================================================

xgb_model.fit(X_fit, y_fit)

# ============================================================
# Cross-Validation Performance
# ============================================================
# Repeated Stratified 5-Fold Cross-Validation
# 5 folds x 5 repeats = 25 validation evaluations.
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=21
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy"
}

cv_results = cross_validate(
    xgb_model,
    X_fit,
    y_fit,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

print("\nCross-Validation Model Performance")
print("=" * 50)

print(f"Accuracy:           {cv_results['test_accuracy'].mean():.4f}")
print(f"Accuracy Std:       {cv_results['test_accuracy'].std():.4f}")
print(f"Precision:          {cv_results['test_precision'].mean():.4f}")
print(f"Recall:             {cv_results['test_recall'].mean():.4f}")
print(f"F1 Score:           {cv_results['test_f1'].mean():.4f}")
print(f"Balanced Accuracy:  {cv_results['test_balanced_accuracy'].mean():.4f}")
print(f"Training Accuracy:  {cv_results['train_accuracy'].mean():.4f}")

# ============================================================
# Generate Validation Probabilities
# ============================================================

val_probs = xgb_model.predict_proba(X_val)[:, 1]

print("\nValidation Probabilities:")
print(val_probs)

# ============================================================
# Evaluate Different Classification Thresholds
# ============================================================

threshold_results = []

thresholds = np.arange(0.30, 0.71, 0.01)

for threshold in thresholds:

    val_pred = (val_probs >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_val, val_pred),
        "Precision": precision_score(
            y_val, val_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, val_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_val, val_pred, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nTop 10 Classification Thresholds by F1")
print("=" * 50)

print(
    threshold_df.sort_values(
        by=["F1 Score", "Accuracy"],
        ascending=False
    ).head(10).to_string(index=False)
)

# ============================================================
# Evaluate Validation Performance at Baseline Threshold
# ============================================================

classification_threshold = 0.50

val_pred = (val_probs >= classification_threshold).astype(int)

print("\nValidation Performance at Threshold 0.50")
print("=" * 50)

print(f"Accuracy:  {accuracy_score(y_val, val_pred):.4f}")
print(f"Precision: {precision_score(y_val, val_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, val_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_val, val_pred, zero_division=0):.4f}")

print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))

print(f"\nBenchmark Classification Threshold: {classification_threshold:.2f}")

# ============================================================
# Retrain Model on Entire Training Set
# ============================================================

xgb_model.fit(X_train, y_train)

# ============================================================
# Generate Final Test Predictions
# ============================================================

y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]

y_pred = (y_pred_probs >= classification_threshold).astype(int)

# ============================================================
# Evaluate Final Model
# ============================================================

test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall = recall_score(y_test, y_pred, zero_division=0)
test_f1 = f1_score(y_test, y_pred, zero_division=0)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\nFinal Test Model Performance")
print("=" * 50)

print(f"Threshold:          {classification_threshold:.2f}")
print(f"Accuracy:           {test_accuracy:.4f}")
print(f"Precision:          {test_precision:.4f}")
print(f"Recall:             {test_recall:.4f}")
print(f"F1 Score:           {test_f1:.4f}")
print(f"Balanced Accuracy:  {test_balanced_accuracy:.4f}")

# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# Confusion Matrix
# ============================================================

test_cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(test_cm)

# ============================================================
# Majority-Class Baseline Comparison
# ============================================================
# Predict the most frequent class in the training data
# for every observation in the test set.
# ============================================================

majority_class = y_train.value_counts().idxmax()

baseline_predictions = np.full(
    shape=len(y_test),
    fill_value=majority_class
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

accuracy_improvement = test_accuracy - baseline_accuracy

print("\nMajority-Class Baseline Comparison")
print("=" * 50)

print(f"Majority Class:       {majority_class}")
print(f"Baseline Accuracy:    {baseline_accuracy:.4f}")
print(f"XGBoost Accuracy:     {test_accuracy:.4f}")
print(f"Accuracy Improvement: {accuracy_improvement:.4f}")

print(
    f"Correct XGBoost Predictions: "
    f"{(y_test.values == y_pred).sum()} out of {len(y_test)}"
)

print(
    f"Correct Baseline Predictions: "
    f"{(y_test.values == baseline_predictions).sum()} out of {len(y_test)}"
)

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
# Feature Importance Analysis and Visualization
# ============================================================

# Extract feature importance from the trained XGBoost model
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

# Sort features by importance in descending order
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print("=" * 40)

print(feature_importance.to_string(index=False))

# ============================================================
# Visualize Feature Importance
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

# Display the most important feature at the top
plt.gca().invert_yaxis()

# Add chart labels and title
plt.title(
    "XGBoost Feature Importance - Happy Customers",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Importance Score", fontsize=11)
plt.ylabel("Survey Features", fontsize=11)

# Add importance values beside each bar
for index, value in enumerate(feature_importance["Importance"]):
    plt.text(
        value + 0.005,
        index,
        f"{value:.4f}",
        va="center",
        fontsize=10
    )

plt.tight_layout()

# Save visualization for GitHub README
plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Final XGBoost Model Parameters
# ============================================================

print("\nFinal XGBoost Model Parameters")
print("=" * 50)

print(f"n_estimators:       {xgb_model.n_estimators}")
print(f"max_depth:          {xgb_model.max_depth}")
print(f"learning_rate:      {xgb_model.learning_rate}")
print(f"min_child_weight:   {xgb_model.min_child_weight}")
print(f"subsample:          {xgb_model.subsample}")
print(f"colsample_bytree:   {xgb_model.colsample_bytree}")
print(f"reg_lambda:         {xgb_model.reg_lambda}")
print(f"scale_pos_weight:   {xgb_model.scale_pos_weight}")
print(f"classification_threshold: {classification_threshold}")


print("\nCurrent Test Benchmark")
print("=" * 40)

print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"Threshold: {classification_threshold:.2f}")
