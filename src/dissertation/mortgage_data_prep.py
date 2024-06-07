import pandas as pd
from common_functions import load_mortgage_data

# Load data (using the common function)
homeMortgage = load_mortgage_data()

# 1. Select the desired columns
selectedCols = [
    "action_taken", "applicant_ethnicity",
    "applicant_income_000s", "applicant_race_1", "co_applicant_ethnicity",
    "co_applicant_sex", "county_code", "hoepa_status", "lien_status",
    "loan_purpose", "loan_type", "msamd",
    "owner_occupancy", "preapproval",
    "property_type", "purchaser_type", "loan_amount_000s"
]

# Create a copy to avoid warnings
homeMortgage_selectedCols = homeMortgage[selectedCols].copy()

# 2. Create target variable (isLoanOriginated)
homeMortgage_selectedCols['isLoanOriginated'] = homeMortgage_selectedCols['action_taken'] == 1

# 3. Remove the original action_taken column
homeMortgage_selectedCols = homeMortgage_selectedCols.drop(columns=[
                                                           'action_taken'])

# 4. Convert columns to categorical types for later modeling
categorical_columns = [
    "applicant_ethnicity", "applicant_race_1", "co_applicant_ethnicity", "co_applicant_sex",
    "county_code", "hoepa_status", "lien_status", "loan_purpose", "loan_type",
    "owner_occupancy", "preapproval", "property_type", "purchaser_type"
]

for col in categorical_columns:
    homeMortgage_selectedCols[col] = homeMortgage_selectedCols[col].astype(
        'category')
