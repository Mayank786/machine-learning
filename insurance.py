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
    #EDA (Exploratory Data Analysis)
    #Data Cleaning and preprocessing
    #to not hamper the real data we will create a copy of the original data frame
df_cleaned = df.copy()
df_cleaned.drop_duplicates(inplace=True)
print(df_cleaned.isnull().sum())
print(df_cleaned.shape)
print(df_cleaned.dtypes)

df_cleaned['sex'].value_counts() #to check the unique values like Male and male as it counts to 2 diff. values
print(df_cleaned['smoker'].value_counts())
print(df_cleaned['sex'].unique())

#Now for for sex column we will convert the values into 0 and 1 where 0 will represent male and 1 will represent female
df_cleaned['sex'] = df_cleaned['sex'].map({'male': 0, 'female': 1})
df_cleaned['smoker'] = df_cleaned['smoker'].map({'yes': 1, 'no': 0})
print(df_cleaned.head())
#Now we will convert the categorical columns into numerical columns using one-hot encoding
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True) #drop_first=True to avoid multicollinearity
print(df_cleaned.head())
df_cleaned=df_cleaned.astype(int)
print(df_cleaned.head())
#Feature engineering and Extraction
#BMI Range 18.5 - 24.9 Healthy Weight; 25.0 - 29.9 Overweight; 30.0 or higher Obese