import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from common_functions import load_mortgage_data, fillColor
from mortgage_data_prep import homeMortgage_selectedCols
import numpy as np

# Evaluate fairness for protected attributes
import fairlearn.metrics as flm

import shap


# Helper function to wrap XGBoost's prediction method
def xgboost_predict(data):
    data_dmatrix = xgb.DMatrix(data)
    return model.predict(data_dmatrix)


# 1. Convert categorical columns to dummy variables (one-hot encoding)
homeMortgage_selectedCols2 = pd.get_dummies(
    homeMortgage_selectedCols, drop_first=True)

# Check if the column exists before attempting to drop it
if 'isLoanOriginated_False' in homeMortgage_selectedCols2.columns:
    homeMortgage_selectedCols2 = homeMortgage_selectedCols2.drop(
        columns=['isLoanOriginated_False'])


# 2. Convert boolean target variable to integer labels
homeMortgage_selectedCols2['isLoanOriginated'] = homeMortgage_selectedCols2['isLoanOriginated'].astype(
    int)

# 3. Splitting Data into Training and Test Sets
X = homeMortgage_selectedCols2.drop("isLoanOriginated", axis=1)
y = homeMortgage_selectedCols2["isLoanOriginated"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=13)

# 4. Prepare Data for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 5. Define XGBoost Parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.05,
    'max_depth': 3,
    'gamma': 0,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'subsample': 1,
    'seed': 13,
    'n_rounds': 100
}

# 6. Train XGBoost Model
model = xgb.train(params, dtrain, num_boost_round=100)

# 7. Make Predictions and Evaluate
y_pred = model.predict(dtest)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# 8. Get Feature Importance and Rank
importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame(list(importance.items()), columns=[
                             'Variables', 'Importance'])

# Sort by importance in descending order and assign rank
importance_df = importance_df.sort_values(
    by='Importance', ascending=False).reset_index(drop=True)
importance_df['Rank'] = importance_df.index + \
    1  # Add 1 to get ranks starting from 1

# Select top 20 features
top_20_features = importance_df.head(20)  # Use head(20) to get the top 20

# 9. Plot Feature Importance with correct rankings
plt.figure(figsize=(10, 8))
plt.barh(top_20_features['Variables'],
         top_20_features['Importance'], color=fillColor)

# Add rank labels to the bars, properly aligned
for i, row in top_20_features.iterrows():
    plt.text(0, i, f"#{int(row['Rank'])}",
             color='black', ha='left', va='center', fontweight='bold')

# Add plot labels and title
plt.xlabel('Importance')
plt.ylabel('Variables')
plt.title('Relative Variable Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Assuming these are the protected attributes
protected_attrs = ['applicant_race_1_5',
                   'co_applicant_ethnicity_2']
for attr in protected_attrs:
    # Demographic Parity Difference
    dpd = flm.demographic_parity_difference(y_true=y_test,
                                            y_pred=predictions,
                                            sensitive_features=X_test[attr])

    # Equal Opportunity Difference
    eod = flm.equalized_odds_difference(y_true=y_test,
                                        y_pred=predictions,
                                        sensitive_features=X_test[attr])

    print(f"\nFairness Metrics for {attr}:")
    print(f"  - Demographic Parity Difference: {dpd:.4f}")
    print(f"  - Equalized Odds Difference: {eod:.4f}")


# 10. SHAP Values for Feature Importance and Interaction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Visualize overall feature importance
plt.figure()
shap.summary_plot(shap_values, X_train, plot_type="bar",
                  show=False)  # Bar plot for feature importance
plt.title("Overall Feature Importance (SHAP Values)")
plt.show()

# Visualize feature importance for each prediction
shap.summary_plot(shap_values, X_train)

# Explore feature interactions
shap_interaction_values = explainer.shap_interaction_values(X_train)
shap.summary_plot(shap_interaction_values, X_train)

# Dependence plot for a specific feature (e.g., 'applicant_income_000s')
shap.dependence_plot('applicant_income_000s', shap_values, X_train)

# Force plot for a single prediction (e.g., the first instance)
shap.initjs()
shap.force_plot(explainer.expected_value,
                shap_values[0, :], X_train.iloc[0, :])

# Visualize impact of protected attributes
for attr in protected_attrs:
    if attr not in X_train.columns:
        print(f"Protected attribute '{attr}' not found in the dataset.")
        continue

    # Set interaction_index to None
    shap.dependence_plot(attr, shap_values, X_train, interaction_index=None)
