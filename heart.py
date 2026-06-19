import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df= pd.read_csv('/workspaces/machine-learning/heart.csv')
# print(df.head())
print(df.shape)
# print(df.info())
# print(df.columns)
# print(df.duplicated().sum())
# print(df.isnull().sum())
# print(df['Cholesterol'].value_counts())
#those having cholesterol level 0 are not valid, so we will replace them with the mean value of the column 
df['Cholesterol'] = df['Cholesterol'].replace(0, df['Cholesterol'].mean())
df['Cholesterol'] = df['Cholesterol'].round(2) # round to 2 decimal places
# print(df['Cholesterol'].value_counts())

# df_cleaned = df.copy()
# df_cleaned.drop_duplicates(inplace=True)
# print(df_cleaned.shape)
print(df['HeartDisease'].value_counts()) # the output shows that there are 508 cases of heart disease and 410 cases of no heart disease in the dataset.
