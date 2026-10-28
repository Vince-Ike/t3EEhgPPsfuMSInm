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
    RandomizedSearchCV)
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
# Define Hyperparameter Search
#=================================================================

param_grid = {

    'n_estimators': [ 100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.08, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [ 1, 3, 5],
    'gamma': [0, 0.1, 0.5]
}
#=================================================================
# Initialize XGBoost Model
#=================================================================
xgb_model= XGBClassifier(
    random_state = 21,
    objective = 'binary:logistic',
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

#=================================================================
# Configure Randomized Search
#=================================================================
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=30,
    scoring='accuracy',
    cv=cv,
    verbose=1,
    random_state=21,
    n_jobs=1
)

#=================================================================
# Train Model
#=================================================================
random_search.fit(X_train, y_train)

# ================================================================
# Display Best Cross Validation
#================================================================
print("\nHyperparameter Optimazation Results")
print("="* 40)

print(f"Best CV Accuracy: {random_search.best_score_:.4f}")
print("\nBest Parameters:")
for parameter, value in random_search.best_params_.items():
    print(f"{parameter}: {value}")
# ============================================================
# Generate Predictions
# ============================================================
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

# ============================================================
#  Evaluate Model Performance
# ============================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1= f1_score(y_test, y_pred, zero_division=0)

print("\nOptimized Model Performance")
print("=" * 40)
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