import pandas as pd
congress = pd.read_csv('congress_cleaned.csv')

df = congress.copy()
target = 'Party'
encode = ['Religion']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Democratic': 0, 'Republican': 1, 'Independent': 2}
def target_encode(val):
    return target_mapper[val]

df['Party'] = df['Party'].apply(target_encode)

# Separating X and y
X = df.drop('Party', axis=1)
Y = df['Party']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('congress_clf.pkl', 'wb'))