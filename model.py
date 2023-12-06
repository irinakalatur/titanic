import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score
import pickle

df = pd.read_csv("train.csv")

def transformation(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df.dropna(inplace=True)
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
model = CatBoostClassifier(class_weights=(1, 2))
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), verbose=False)
pred = model.predict(X_test)
print(pred)
pred_prob = model.predict_proba(X_test)
print(X_test)
print(pred_prob)
rs = recall_score(y_test, pred)
print("Recall score:", rs)
file_model = 'model.pkl'
pickle.dump(model, open(file_model, 'wb'))