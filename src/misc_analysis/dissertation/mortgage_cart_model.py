import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from mortgage_data_prep import homeMortgage_selectedCols

# 1. Split the Data into Training and Test Sets
X = homeMortgage_selectedCols.drop("isLoanOriginated", axis=1)
y = homeMortgage_selectedCols["isLoanOriginated"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=3000
)  # 70% training and 30% test

# 2.Encode Categorical Variables
ordinal_encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value', unknown_value=-1)
X_train[X_train.select_dtypes(include=['category']).columns] = ordinal_encoder.fit_transform(
    X_train.select_dtypes(include=['category']))
X_test[X_test.select_dtypes(include=['category']).columns] = ordinal_encoder.transform(
    X_test.select_dtypes(include=['category']))


# 3. Build the CART Model
clf = DecisionTreeClassifier(
    min_samples_leaf=5, random_state=3000
)  # min_samples_leaf is similar to minbucket in R
clf.fit(X_train, y_train)

# 4. Predict on the Test Set
y_pred = clf.predict(X_test)

# 5. Evaluate the Model (Confusion Matrix)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# 6. Visualize the Tree
plt.figure(figsize=(20, 10))  # Make the figure larger for better readability
plot_tree(
    clf,
    filled=True,
    feature_names=X.columns,
    class_names=["Not Originated", "Originated"],
)
plt.show()
