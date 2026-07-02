import numpy as np
import pandas as pd
df = pd.read_csv('/workspaces/machine-learning/Feature_scaling/covid_toy.csv')
#print(df.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['has_covid']),df['has_covid'],test_size=0.2)                                                


from math import remainder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
transformer = ColumnTransformer(transformers=[
    ('tnf1',SimpleImputer(),['fever']),
    ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
    ('tnf3',OneHotEncoder(sparse_output=False,drop='first'),['gender','city'])
],remainder='passthrough') #reminder can have drop or passthrough means I don't want to do anything on leftover columns(age)

transformer.fit(X_train) # Fit the transformer first

X_train_transformed = transformer.transform(X_train)
X_test_transformed = transformer.transform(X_test)

# Get the feature names after transformation
feature_names = transformer.get_feature_names_out()

X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names) #columns=feature_names, so that coulums should be visible 
X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)

print(X_train_transformed.head())
print(X_test_transformed.head())