# ============================================================
# XGBoost Classification Model
# ============================================================
# Purpose:
# Prepare the dataset
# train an XGBoost classifier, and evaluate model
# performance using classification metrics.
# ============================================================
import pandas as pd
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    cross_val_score)
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

df = pd.read_csv("/Users/vinceike/Desktop/Apziva Projects/ACME-HappinessSurvey2020.csv")
df.shape
#print(df.isnull().sum())
# no missing data nor different types of data
#==============================================================
# Split features and target columns
#==============================================================
X = df.drop(columns=['Y'])
y = df['Y']
#==============================================================
# Split data into Training and Testing
#===============================================================
# 80/20 Train/Test split
# stratifying y
X_train, X_test, y_train, y_test = train_test_split(
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
    subsample = 0.8,
    random_state = 21,
    eval_metric ='logloss'

)
# ==============================================================
# Use 5 Fold Cross Validation
# ===============================================================
cv = StratifiedKFold(
    n_splits= 5,
    shuffle=True,
    random_state=21
)

cv_scores = cross_val_score(
    xgb_model,
    X_train,
    y_train,
    cv=cv,
    scoring='accuracy'
)

print("\n5-Fold Cross Validation Results")
print("-----------------------------------")
for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold} Accuracy: {score:.4f}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")
#=================================================================
# Train Model
#=================================================================
xgb_model.fit(X_train, y_train)


# ============================================================
# Generate Predictions
# ============================================================
y_pred = xgb_model.predict(X_test)

# ============================================================
#  Evaluate Model Performance
# ============================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1= f1_score(y_test, y_pred, zero_division=0)

print("\nModel Performance")
print("-------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ============================================================
# Classification Report
# ============================================================

# Display precision, recall, F1 score, and support
# for each target class.

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)