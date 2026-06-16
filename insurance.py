import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df= pd.read_csv('/workspaces/machine-learning/insurance.csv')
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.columns)
numeric_columns = ['age', 'bmi', 'children','charges']
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()