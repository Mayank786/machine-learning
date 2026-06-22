import numpy as np
import pandas as pd
df = pd.read_csv('/content/cars.csv')
# print(df.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:4],df.iloc[:,-1],test_size=0.2,random_state=2)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first',sparse=False,dtype=np.int32)
X_train_new = ohe.fit_transform(X_train[['fuel','owner']])
X_test_new = ohe.transform(X_test[['fuel','owner']])

np.hstack((X_train[['brand','km_driven']].values,X_train_new)) #to join all colums

#OneHotEncoding with Top Categories

counts = df['brand'].value_counts()
df['brand'].nunique()
threshold = 100
repl = counts[counts <= threshold].index
print(pd.get_dummies(df['brand'].replace(repl, 'uncommon')).sample(5))

