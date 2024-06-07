import pandas as pd
import matplotlib.pyplot as plt
from common_functions import load_mortgage_data, fillColor
import seaborn as sns

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

# 1. Join and calculate percentages (add to existing script)
homeMortgageStatus_race = (pd.merge(homeMortgageStatus_applicant_race1, homeMortgage_applicant_race1, on='applicant_race_name_1')
                           .assign(percentage=lambda df: (df['CountOfActionTaken'] / df['CountOfRace1']) * 100))

# 2. Sort for consistent facet order
top_action = homeMortgageStatus_race['action_taken_name'].value_counts(
).index[0]
facet_order = (homeMortgageStatus_race[homeMortgageStatus_race['action_taken_name'] == top_action]
               .sort_values('percentage', ascending=False)['applicant_race_name_1'].tolist())

# 3. Plot with FacetGrid, horizontal bars, and shared y-axis label
g = sns.FacetGrid(homeMortgageStatus_race, col='applicant_race_name_1', col_wrap=3,
                  sharey=True, height=5, aspect=1.5, col_order=facet_order)  # Share y-axis
g.map(sns.barplot, 'percentage', 'action_taken_name',
      order=homeMortgageStatus_race['action_taken_name'].unique(), color=fillColor)

# Add percentage labels
for ax in g.axes.flat:
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.text(width + 1, p.get_y() + p.get_height() / 2, f"{width:.0f}%",
                    color='black', ha='left', va='center', fontweight='bold')

# Main title and labels
# This line replaces the default title template
g.set_titles(col_template="{col_name}")

g.fig.suptitle('Actions in Loans by Race', y=1.02, fontsize=16)
g.set_axis_labels("%age Count Of Action Taken",
                  "Action")  # Shared y-axis label

# Show the plot
plt.tight_layout()
plt.show()
