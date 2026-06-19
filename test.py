import pandas as pd
from ydata_profiling import ProfileReport
df = pd.read_csv("/workspaces/machine-learning/Titanic-Dataset.csv")
profile = ProfileReport(df)
profile.to_file("output.html")