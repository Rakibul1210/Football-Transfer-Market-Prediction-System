import pandas as pd
from sklearn.preprocessing import OneHotEncoder


data_path = 'data\data_22-23.csv'

df = pd.read_csv(data_path)
# print(df)

nations = df["Pos"]
print(nations.value_counts())

# nations = pd.get_dummies(nations)
# print(nations)


encoder = OneHotEncoder()
result = encoder.fit_transform(df).toarray()
print(result)