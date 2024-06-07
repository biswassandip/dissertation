import pandas as pd
import matplotlib.pyplot as plt
from common_functions import load_mortgage_data, fillColor2
import textwrap
import seaborn as sns

# Load data (using the common function)
homeMortgage = load_mortgage_data()


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

# 3. Join and calculate percentages
homeMortgageStatus_ethnicity2 = (pd.merge(homeMortgageStatus_ethnicity, homeMortgage_ethnicity, on='applicant_ethnicity_name')
                                 .assign(percentage=lambda df: (df['CountOfActionTaken'] / df['CountOfEthnicity']) * 100))


# 4. Plotting with FacetGrid and wrapped labels

# Sort the data by the average percentage of the top action to determine the column order
top_action = homeMortgageStatus_ethnicity2['action_taken_name'].value_counts(
).index[0]
facet_order = homeMortgageStatus_ethnicity2[homeMortgageStatus_ethnicity2['action_taken_name'] == top_action].sort_values(
    'percentage', ascending=False)['applicant_ethnicity_name'].tolist()

# Increased figure height and width to improve readability
g = sns.FacetGrid(homeMortgageStatus_ethnicity2, col='applicant_ethnicity_name', col_wrap=3,
                  sharey=False, height=5, aspect=1.5, col_order=facet_order)  # Adjust col_wrap and aspect as needed

# Create the bar plots on each facet
g.map(sns.barplot, 'percentage', 'action_taken_name',
      order=homeMortgageStatus_ethnicity2['action_taken_name'].unique(), color=fillColor2)

# Wrap and set y-axis labels (action_taken_name)
for ax in g.axes.flat:
    # Adjust wrap width (15) as needed
    labels = [textwrap.fill(label.get_text(), 15)
              for label in ax.get_yticklabels()]
    ax.set_yticklabels(labels)

# Wrap and set column titles (applicant_ethnicity_name)
for ax in g.axes.flat:
    title = ax.get_title()
    # Adjust wrap width (20) as needed
    wrapped_title = textwrap.fill(title, 20)
    ax.set_title(wrapped_title)

# Add percentage labels to the bars
for ax in g.axes.flat:
    for p in ax.patches:
        width = p.get_width()
        if width > 0:  # Only label bars with a positive value
            ax.text(width + 1, p.get_y() + p.get_height() / 2, f"{width:.0f}%",
                    color='black', ha='left', va='center', fontweight='bold')

# Add titles and labels
g.fig.suptitle(textwrap.fill('Actions in Loans by Ethnicity', 60),
               y=1.02, fontsize=16)  # Wrap the main title
# Remove y-label since it's now in the column titles
g.set_axis_labels('%age Count Of Action Taken', '')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()
