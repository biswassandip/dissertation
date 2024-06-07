import pandas as pd
import matplotlib.pyplot as plt
from common_functions import load_mortgage_data, fillColor

# Load data (using the common function)
homeMortgage = load_mortgage_data()

# 1. Analyze action taken by applicant's primary race
homeMortgageStatus_applicant_race1 = (homeMortgage.groupby(['action_taken_name', 'applicant_race_name_1'])
                                                  .size()
                                                  .reset_index(name='CountOfActionTaken')
                                                  .sort_values(by='CountOfActionTaken', ascending=False))

# 2. Analyze overall distribution of applicant's primary races
homeMortgage_applicant_race1 = (homeMortgage.groupby('applicant_race_name_1')
                                .size()
                                .reset_index(name='CountOfRace1')
                                .sort_values(by='CountOfRace1', ascending=False))

# 3. Plot the distribution of applicant's primary races
plt.figure(figsize=(10, 8))  # Optional: Adjust plot size

plt.barh(homeMortgage_applicant_race1['applicant_race_name_1'],
         homeMortgage_applicant_race1['CountOfRace1'], color=fillColor)

# Add text labels to the bars
for i, v in enumerate(homeMortgage_applicant_race1['CountOfRace1']):
    plt.text(v + 1, i, f"({v})", color='black', va='center', fontweight='bold')

# Add plot labels and title
plt.xlabel('Count Of Action Taken')
plt.ylabel('Race Name')
plt.title('Actions in Loans by Race')
plt.gca().invert_yaxis()  # Flip the y-axis to match the R plot

plt.show()
