import pandas as pd
import matplotlib.pyplot as plt
from common_functions import load_mortgage_data, fillColor2

# Load the data (make sure this path matches your file location)
homeMortgage = load_mortgage_data()  # Load data using the function

# 1. Analyze action taken by ethnicity
homeMortgageStatus_ethnicity = (homeMortgage.groupby(['action_taken_name', 'applicant_ethnicity_name'])
                                            .size()
                                            .reset_index(name='CountOfActionTaken')
                                            .sort_values(by='CountOfActionTaken', ascending=False))

# 2. Analyze overall distribution of ethnicities
homeMortgage_ethnicity = (homeMortgage.groupby('applicant_ethnicity_name')
                          .size()
                          .reset_index(name='CountOfEthnicity')
                          .sort_values(by='CountOfEthnicity', ascending=False))

# 3. Plot the distribution of ethnicities
plt.figure(figsize=(10, 8))  # Optional: Adjust plot size

plt.barh(homeMortgage_ethnicity['applicant_ethnicity_name'],
         homeMortgage_ethnicity['CountOfEthnicity'], color=fillColor2)

# Add text labels to the bars
for i, v in enumerate(homeMortgage_ethnicity['CountOfEthnicity']):
    plt.text(v + 1, i, f"({v})", color='black', va='center', fontweight='bold')

plt.xlabel('Count Of Applicants')
plt.ylabel('applicant_ethnicity_name')
plt.title('Distribution of Applicant Ethnicities')
plt.gca().invert_yaxis()
plt.show()
