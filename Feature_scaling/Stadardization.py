import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_csv("/workspaces/machine-learning/Feature_scaling/Social_Network_Ads.csv")

# print("Shape")
# print(df.shape)

# print("\nInfo")
# print(df.info())

# print("\nMissing Values")
# print(df.isnull().sum())

# print("\nDuplicates")
# print(df.duplicated().sum())

# print("\nStatistics")
# print(df.describe())

# print("\nCategorical Statistics")
# print(df.describe(include='object'))
# print(df.head())
df=df.iloc[:,2:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Purchased', axis=1), df['Purchased'], test_size=0.3, random_state=0) #This will split the data into 70% training and 30% testing, with a fixed random state for reproducibility.
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #creating scaler object, which will be used to standardize the features by removing the mean and scaling to unit variance. This is a common preprocessing step in machine learning to ensure that all features are on the same scale, which can improve the performance of many algorithms.

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train) #This line fits the scaler to the training data, which means it calculates the mean and standard deviation for each feature in the training set. These parameters will be used to transform both the training and test sets, ensuring that the same scaling is applied to both datasets.

# transform train and test sets
X_train_scaled = scaler.transform(X_train) #This line applies the scaling transformation to the training data using the parameters learned from the previous fit step. It standardizes the features in the training set by removing the mean and scaling to unit variance.
X_test_scaled = scaler.transform(X_test) #This line applies the same scaling transformation to the test data using the parameters learned from fitting the scaler to the training data. This ensures that the test set is scaled in the same way as the training set, which is important for maintaining consistency and ensuring that the model can make accurate predictions on the test data.
# print(scaler.mean_)

#Important: 
#Now we above gave X_train as df but X_train_scaled return as numpy array. So, we need to convert it back to dataframe.
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
print(X_train_scaled.describe().round(2))
print(X_test_scaled.describe().round(2))