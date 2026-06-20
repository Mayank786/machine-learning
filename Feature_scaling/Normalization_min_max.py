import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df= pd.read_csv('/workspaces/machine-learning/Feature_scaling/wine_data.csv',header=None,usecols=[0,1,2])
df.columns=['Class label', 'Alcohol', 'Malic acid']

# Create a processed DataFrame by dropping the first row (the old header)
# and converting all relevant columns to numeric types.
df_processed = df.iloc[1:].copy()
df_processed['Class label'] = pd.to_numeric(df_processed['Class label'])
df_processed['Alcohol'] = pd.to_numeric(df_processed['Alcohol'])
df_processed['Malic acid'] = pd.to_numeric(df_processed['Malic acid'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_processed.drop('Class label', axis=1),df_processed['Class label'],test_size=0.3,random_state=0) #This will split the data into 70% training and 30% testing, with a fixed random state for reproducibility.
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Important: 
#Now we above gave X_train as df but X_train_scaled return as numpy array. So, we need to convert it back to dataframe.
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(X_train_scaled.describe().round(1))
print(X_test_scaled.describe().round(1))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(X_train['Alcohol'], X_train['Malic acid'],c=y_train)
ax1.set_title("Before Scaling")
ax2.scatter(X_train_scaled['Alcohol'], X_train_scaled['Malic acid'],c=y_train)
ax2.set_title("After Scaling")
plt.show()




                                                    

