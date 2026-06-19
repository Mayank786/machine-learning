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
# print(df_cleaned.isnull().sum())
# print(df_cleaned.shape)
# print(df_cleaned.dtypes)

df_cleaned['sex'].value_counts() #to check the unique values like Male and male as it counts to 2 diff. values
# print(df_cleaned['smoker'].value_counts())
# print(df_cleaned['sex'].unique())

#Now for for sex column we will convert the values into 0 and 1 where 0 will represent male and 1 will represent female
df_cleaned['sex'] = df_cleaned['sex'].map({'male': 0, 'female': 1})
df_cleaned['smoker'] = df_cleaned['smoker'].map({'yes': 1, 'no': 0})
# print(df_cleaned.head())

#Now we will convert the categorical columns into numerical columns using one-hot encoding
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True) #drop_first=True to avoid multicollinearity
# print(df_cleaned.head())
df_cleaned=df_cleaned.astype(int) #to convert all the columns into integer type
# print(df_cleaned.head())

#Feature engineering and Extraction

#BMI Range 0-18.5: Underweight, 18.5-24.9: Normal, 25-29.9: Overweight, 30 and above: Obese

#making a new column called bmi_category based on the bmi values
#pd.cut() is a function in pandas that is used to segment and sort data values into bins or categories. It is often used for creating categorical variables based on continuous data. In this case, we are using pd.cut() to categorize the 'bmi' column into four categories: 'Underweight', 'Normal', 'Overweight', and 'Obese', based on the specified bins.
df_cleaned['bmi_category'] = pd.cut(df_cleaned['bmi'],bins=[0, 18.5, 24.9, 29.9, float('inf')], #float('inf') is used to represent positive infinity, which means any value greater than 29.9 will be categorized as 'Obese'.
labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
# print(df_cleaned.head())

#Now we will convert the bmi into numerical values using one-hot encoding
df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
df_cleaned=df_cleaned.astype(int) #to convert all the columns into integer type
# print(df_cleaned.head())

#Now we'll do FEATURE SCALING to bring all the features to the same scale, which is important for many machine learning algorithms. We'll use StandardScaler from sklearn for this purpose.

#we will scale the features 'age', 'bmi', 'children' not 'charges'
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #StandardScaler standardizes features by removing the mean and scaling to unit variance. The formula is: z = (x - u) / s, where u is the mean of the training samples, and s is the standard deviation of the training samples.
df_cleaned[['age', 'bmi', 'children']] = scaler.fit_transform(df_cleaned[['age', 'bmi', 'children']])
print(df_cleaned.head())

#Now from these columns we'll do EXTRACTION using Pearson correlation coefficient to find the correlation between the features and the target variable 'charges'. The Pearson correlation coefficient is a measure of the linear correlation between two variables, giving a value between +1 and -1 inclusive, where 1 is total positive linear correlation, 0 is no linear correlation, and -1 is total negative linear correlation.
from scipy.stats import pearsonr

# List of features to check against target
selected_features = ['age', 'bmi', 'children', 'sex', 'smoker','region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese']

correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0] #to calculate the Pearson correlation coefficient between each feature and the target variable 'charges'. The pearsonr function returns a tuple where the first element is the correlation coefficient and the second element is the p-value. We are only interested in the correlation coefficient, hence we use [0] to extract it.
    for feature in selected_features
}
correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Pearson Correlation']) #to create a dataframe from the correlations dictionary, where 'Feature' is the name of the feature and 'Pearson Correlation' is the corresponding correlation value.
correlation_df.sort_values(by='Pearson Correlation', ascending=False)
print(correlation_df.sort_values(by='Pearson Correlation', ascending=False))

#p_value is a measure of the evidence against a null hypothesis. A low p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis. A high p-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis. In this context, a low p-value would suggest that there is a statistically significant correlation between the feature and the target variable 'charges'.
# p_values and chi2 value is same as p_value but it is used for categorical variables and chi2 value is used to measure the association between two categorical variables. A high chi2 value indicates a strong association between the variables, while a low chi2 value indicates a weak association. In this context, a high chi2 value would suggest that there is a strong association between the feature and the target variable 'charges'.

from scipy.stats import chi2_contingency

cat_features = ['sex', 'smoker','region_northwest', 'region_southeast', 'region_southwest','bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'] #not mentioned charges because it is the target variable and we are trying to find the association between the features and the target variable

alpha = 0.05

df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False) # Create bins for the target variable

chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin']) #to create a contingency table (cross-tabulation) between the categorical feature (col) and the binned target variable ('charges_bin'). The pd.crosstab function computes a simple cross-tabulation of two (or more) factors. In this case, it will count the occurrences of each combination of the categorical feature and the binned target variable, resulting in a contingency table that can be used for the chi-squared test.
    chi2_stat, p_val, _, _ = chi2_contingency(contingency) #to perform the chi-squared test of independence on the contingency table. The chi2_contingency function returns several values, but we are only interested in the chi-squared statistic (chi2_stat) and the p-value (p_val), which are used to determine the strength of the association between the categorical feature and the target variable.
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T #to create a DataFrame from the chi2_results dictionary, where each row corresponds to a categorical feature and the columns contain the chi-squared statistic, p-value, and the decision on whether to keep or drop the feature based on the p-value.
chi2_df = chi2_df.sort_values(by='p_value')
print(chi2_df)

final_df = df_cleaned[['age', 'sex', 'bmi', 'children', 'smoker', 'charges','region_southeast','bmi_category_Obese']]
print(final_df.head())
#save the final dataframe to a new csv file
final_df.to_csv('insurance_cleaned.csv', index=False)
