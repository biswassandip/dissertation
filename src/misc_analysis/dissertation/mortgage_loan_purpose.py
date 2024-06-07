import pandas as pd
import matplotlib.pyplot as plt
from common_functions import load_mortgage_data, fillColor2


# Load data (using the common function)
homeMortgage = load_mortgage_data()

# 1. Filter and aggregate data
loan_purpose_counts = (homeMortgage.dropna(subset=['loan_purpose_name'])  # Remove rows with missing loan purpose
                       .groupby('loan_purpose_name')
                       .size()
                       .reset_index(name='CountLoanPurpose')
                       .assign(percentage=lambda df: (df['CountLoanPurpose'] / df['CountLoanPurpose'].sum()) * 100)
                       .sort_values(by='percentage', ascending=True))  # Sort for plot

# 2. Create horizontal bar plot with percentage labels
plt.figure(figsize=(10, 6))  # Optional: Adjust plot size

plt.barh(loan_purpose_counts['loan_purpose_name'],
         loan_purpose_counts['percentage'], color=fillColor2)

# Add percentage labels
for i, v in enumerate(loan_purpose_counts['percentage']):
    plt.text(v + 1, i, f"({v:.0f}%)", color='black',
             va='center', fontweight='bold')

# Add plot labels and title
plt.xlabel('Percentage')
plt.ylabel('Loan Purpose Type')
plt.title('Loan Purpose Types')

# Show the plot
plt.tight_layout()
plt.show()
