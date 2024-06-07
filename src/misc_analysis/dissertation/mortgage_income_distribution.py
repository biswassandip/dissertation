import pandas as pd
import matplotlib.pyplot as plt
from common_functions import load_mortgage_data

# Load data (using the common function)
homeMortgage = load_mortgage_data()

# Filter for loan originations
actionStatus = "Loan originated"
originated_loans = homeMortgage[homeMortgage['action_taken_name']
                                == actionStatus]

# Set breaks for x-axis
breaks = list(range(0, 401, 50))  # Range from 0 to 400 with increments of 50

# Plot histogram
plt.figure(figsize=(10, 6))  # Optional: Adjust plot size
plt.hist(originated_loans['applicant_income_000s'],
         bins=breaks, color='blue', edgecolor='white')

# Add labels and title
plt.xlabel('Income in Thousands')
plt.ylabel('Count')
plt.title('Loan Originated Applicant Income Distribution')
plt.xticks(breaks)  # Set x-ticks to match the breaks

# Optional: Customize the grid
plt.grid(axis='y', alpha=0.75)

# Show the plot
plt.show()
