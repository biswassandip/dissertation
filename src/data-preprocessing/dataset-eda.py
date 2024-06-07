# libraries
import datetime

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# load the data
loan_data = pd.read_csv(
    "/Users/gini/MSc_AI/Dissertation_Project/dissertation/src/data-input-files/loan-train.csv")
df_train = loan_data.copy()
print(df_train.head())

# Size Of Data Set
print(df_train.shape)

# Columns Names
print(df_train.columns)

# Columns Types
print(df_train.dtypes)

# Info
print(df_train.info())

# Duplicated data
print(df_train[df_train.duplicated() == True])

# Generate descriptive statistics
print(df_train.describe().T)

# Remove any missing valur or data
df_train.isnull().values.any()
df_train.isnull().sum()


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()
               * 100).sort_values(ascending=False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


missing_data(df_train)

df_train_clean = df_train.copy()
df_train_clean = df_train_clean.dropna()
df_train_clean.drop(['Loan_ID'], inplace=True, axis=1)
df_train_clean.info()

{column: list(df_train_clean[column].unique())
 for column in df_train_clean.select_dtypes('object').columns}


# gender distribution
df_gender = df_train_clean['Gender'].value_counts().to_frame().reset_index().rename(columns={'index': 'Gender', 'Gender': 'count'})

fig = go.Figure([go.Pie(labels=df_gender['Gender'],values=df_gender['count'], pull=[0, 0.2], hole=0.4)])
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent',textfont_size=12, insidetextorientation='radial')
fig.update_layout(title="Gender Count", title_x=0.5)
fig.show()
