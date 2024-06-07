import pandas as pd
import matplotlib.pyplot as plt
from common_functions import load_mortgage_data, fillColor


# Optional: Decision tree visualization (choose either method)
# from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
# # Example dataset if you want to test decision tree visualization
# from sklearn.datasets import load_iris
# import graphviz

# 1. Load the mortgage data
homeMortgage = load_mortgage_data()  # Load data using the function

# 2. Analyze and aggregate action taken status
homeMortgageStatus = (homeMortgage.groupby('action_taken_name')
                      .size()  # Count of each action
                      .reset_index(name='CountOfActionTaken')
                      .assign(PercentageActionTaken=lambda df: df['CountOfActionTaken'] / df['CountOfActionTaken'].sum() * 100)
                      .sort_values(by='PercentageActionTaken', ascending=False))

# 3. Plot the distribution of actions taken
plt.figure(figsize=(10, 8))  # Optional: Adjust plot size
plt.barh(homeMortgageStatus['action_taken_name'],
         homeMortgageStatus['PercentageActionTaken'], color=fillColor)

# Add text labels to the bars
for i, v in enumerate(homeMortgageStatus['PercentageActionTaken']):
    plt.text(v + 1, i, f"({v:.2f}%)", color='black',
             va='center', fontweight='bold')

plt.xlabel('Percentage Count Of Action Taken')
plt.ylabel('action_taken_name')
plt.title('Actions in Loans')
plt.gca().invert_yaxis()
plt.show()


# 4. Optional: Decision tree visualization (uncomment and modify as needed)
# Load example data and fit a decision tree (remove or modify as per your dataset)
# iris = load_iris()
# X = iris.data[:, 2:]  # Petal length and width
# y = iris.target
# clf = DecisionTreeClassifier(max_depth=2)
# clf.fit(X, y)

# # Option 1: Using graphviz (install graphviz library and software if you want to use this option)
# dot_data = export_graphviz(clf, out_file=None, 
#                             feature_names=iris.feature_names[2:],  
#                             class_names=iris.target_names,
#                             filled=True, rounded=True,
#                             special_characters=True)

# graph = graphviz.Source(dot_data)  
# graph.view()  # This will open the tree in a separate viewer

# Option 2: Simpler plot using matplotlib
# plt.figure(figsize=(12, 8))
# plot_tree(clf, filled=True, feature_names=iris.feature_names[2:], class_names=iris.target_names)
# plt.show()
