import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

def transformation(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df.dropna(inplace=True)
    df.loc[df['Age'] <= 15, 'Age'] = 0
    df.loc[(df['Age'] > 15) & (df['Age'] <= 25), 'Age'] = 1
    df.loc[(df['Age'] > 25) & (df['Age'] <= 40), 'Age'] = 2
    df.loc[(df['Age'] > 40) & (df['Age'] <= 60), 'Age'] = 3
    df.loc[df['Age'] > 60, 'Age'] = 4
    df['Age'] = df['Age'].astype(int)
    encoder = LabelEncoder()
    df['Sex'] = encoder.fit_transform(df['Sex'])
    df['Embarked'] = encoder.fit_transform(df['Embarked'])
    df = df[['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
    return df


df = transformation(df)
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

cat_features = list(range(0,X.shape[1]))
model = CatBoostClassifier()
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), verbose=False)
pred = model.predict(X_test)
rs = recall_score(y_test, pred)
print("Recall score:", rs)
file_model = 'model.pkl'
pickle.dump(model, open(file_model, 'wb'))