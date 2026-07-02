import numpy as np
import pandas as pd

df = pd.read_csv('Feature_scaling/customer.csv')

df = df.iloc[:,2:] 
print(df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:2], df.iloc[:, -1], test_size=0.2, random_state=0) #1st df.iloc for X and 2nd df.iloc for y


#FOR ORDINAL CATEGORICAL DATA
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']]) #Here we are specifying the order of the categories for each column. For the first column, 'Poor' will be encoded as 0, 'Average' as 1, and 'Good' as 2. For the second column, 'School' will be encoded as 0, 'UG' as 1, and 'PG' as 2.

oe.fit(X_train)

X_train_encoded = oe.transform(X_train)
X_test_encoded = oe.transform(X_test)

X_train_encoded = pd.DataFrame(X_train_encoded, columns=X_train.columns)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=X_test.columns)

print(X_train_encoded.head())
print(X_test_encoded.head())

