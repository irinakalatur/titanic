import pandas as pd
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# загрузка данных
df = pd.read_csv('train.csv')

# создаем подробный отчет по данным и сохраняем в файл
profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("report.html")

# удаляем неинформативные столбцы
df = df.drop(['Name', 'Cabin', 'Ticket'], axis=1)
print(df.head())

# создаем новый столбец с возрастом пассажиров и преобразуем его
# заменяем все NaN средними значениями
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['new_age'] = df['Age']
df['new_age'] = df['new_age'].map(lambda x: int(x // 10))
print(df['new_age'].value_counts())
new_age_survived = df.groupby(['new_age', 'Survived']).size().unstack().fillna(0)
print(new_age_survived)
df.dropna(inplace=True)
print(df.info())


# найдем процент выживших и не выживших
def age_percent(row):
    return [float(col) / sum(row) for col in row][0]

''' Выводим статистику выживших по возрасту.
    Пассажиры в возрасте 15-20 лет, 65-40 лет и 60 лет имеют меньший шанс на выживание, 
    пассажир в возрасте 70 лет не выжил, пассажир в возрасте 80 лет выжил.
'''
new_age_survived = pd.DataFrame(new_age_survived.apply(age_percent, axis=1))
new_age_survived[1] = 1 - new_age_survived[0]
print(new_age_survived)
new_age_survived.plot.barh(stacked=True, color=['#9370DB', '#4B0082'])
plt.show()

''' Выводим статистику выживших по классу.
    Из графика видно, что наибольшее число выживших находится в первом классе.
'''
pclass_survived = df.groupby(['Pclass', 'Survived']).size().unstack()
pclass_survived = pd.DataFrame(pclass_survived.apply(age_percent, axis=1))
pclass_survived[1] = 1 - pclass_survived[0]
print(pclass_survived)
pclass_survived.plot.barh(stacked=True, color=['#9370DB', '#4B0082'])
plt.show()

''' Выводим статистику выживших по порту отправления.
    Наибольшее число выживших отправилось из порта C (Шербург).
'''
embarked_survived = df.groupby(['Embarked', 'Survived']).size().unstack()
embarked_survived = pd.DataFrame(embarked_survived.apply(age_percent, axis=1))
embarked_survived[1] = 1 - embarked_survived[0]
print(embarked_survived)
embarked_survived.plot.barh(stacked=True, color=['#9370DB', '#4B0082'])
plt.show()

''' Выводим статистику выживших по количеству родителей/детей.
    Пассажиры, путешествующие в одиночку и семьи из 3 человек имеют 
    наибольший шанс на выживание, семьи от 4 и больше человек имели 
    совсем низкий шанс на выживание.
'''
sns.barplot(x='Parch', y='Survived', errorbar=None, data=df)
plt.show()

''' Выводим статистику выживших по количеству братьев и сестер/супругов.
    Пассажиры, имеющие 1 брата/сестру имеют высокий шанс на выживание,
    от 4 и выше - очень низкий шанс на выживание.
'''
sns.barplot(x='SibSp', y='Survived', errorbar=None, data=df)
plt.show()

''' Выводим статистику выживших по полу.
    Из графика видно, что выживших женщин гораздо больше, чем мужчин.
'''
sns.barplot(x='Sex', y='Survived', errorbar=None, data=df)
plt.show()

''' Посчитаем процент женщин и мужчин в разных классах.
    В первом и втором классе больше женщин, чем мужчин, 
    большинство мужчин и женщин предпочли третий класс, 
    в третьем классе больше мужчин, чем женщин.
'''
s = pd.DataFrame(df.groupby(['Sex', 'Pclass'], as_index=False)['PassengerId'].count())
f = s[s.Sex == 'female'].copy(deep=True)
f['Percent'] = f.PassengerId/f.PassengerId.sum()
print(f)
m = s[s.Sex == 'male'].copy(deep=True)
m['Percent'] = m.PassengerId/m.PassengerId.sum()
print(m)

''' Смотрим, как распределены женщины и мужчины по классам 
    и построим график выживаемости.
    Больше выживших среди мужчин в 1 классе (около 40%), во 2 и 3 примерно одинаково низкая (около 18%),
    среди женщин так же больше выживших в 1 классе (около 95%), во 2 чуть меньше (около 90%), в 3 около 45%.
'''
sns.catplot(data=df, x='Sex', y='Survived', col='Pclass', kind='bar')
plt.show()

''' Строим распределение по тарифу.
    Большее количество женщин и мужчин предпочли более дешевый тариф,
    очень малое количество женщин и мужчин предпочли тариф дороже,
    самые дорогие тарифы у небольшого количества женщин.
'''
sns.kdeplot(data=df, x='Fare', hue='Sex', fill=True)
plt.show()

''' Строим график выживаемости в зависимости от стоимости тарифа.
    Большое количество пассажиров с дешевым тарифом имеют низкий шанс на выживание,  
    все пассажиры с дорогим тарифом выжили.
'''
sns.kdeplot(data=df, x='Fare', hue='Survived', fill=True)
plt.show()

''' Смотрим на стоимость билетов в каждом классе.
    Самые дорогие билеты в первом классе, следовательно,
    пассажиры первого класса с самыми дорогими билетами 
    имеют большие шансы на выживание.
'''
sns.stripplot(x='Pclass', y='Fare', data=df)
plt.show()

encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df['Embarked'] = encoder.fit_transform(df['Embarked'])


df.loc[df['Age'] <= 15, 'Age'] = 0
df.loc[(df['Age'] > 15) & (df['Age'] <= 25), 'Age'] = 1
df.loc[(df['Age'] > 25) & (df['Age'] <= 40), 'Age'] = 2
df.loc[(df['Age'] > 40) & (df['Age'] <= 60), 'Age'] = 3
df.loc[df['Age'] > 60, 'Age'] = 4
df['Age'] = df['Age'].astype(int)


df = df.drop(['PassengerId', 'new_age', 'Fare'], axis=1)


print(df.head())


# разделяем данные на тестовую и тренировочную части
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Логистическая регрессия
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# строим матрицу ошибок
lr_matrix = confusion_matrix(y_test, lr_pred)
true_names = ['True Survived', 'True Not Survived']
pred_names = ['Predicted Survived', 'Predicted Not Survived']
df_lr_matrix = pd.DataFrame(lr_matrix, index=true_names, columns=pred_names)
sns.heatmap(df_lr_matrix, annot=True, fmt='d')
plt.show()

print(classification_report(y_test, lr_pred))

lr_pred_prob = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, lr_pred_prob)
roc_auc = auc(fpr, tpr)
print('ROC_AUC:', roc_auc)
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC-curve')
plt.show() 

# Решающие деревья
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)

# строим матрицу ошибок
d_matrix = confusion_matrix(y_test, clf_pred)
df_d_matrix = pd.DataFrame(d_matrix, index=true_names, columns=pred_names)
sns.heatmap(df_d_matrix, annot=True, fmt='d')
plt.show()

print(classification_report(y_test, clf_pred))

clf_pred_prob = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, clf_pred_prob)
roc_auc = auc(fpr, tpr)
print('ROC_AUC:', roc_auc)
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC-curve')
plt.show()

# Градиентный бустинг
cat_features = list(range(0,X.shape[1]))
cbcf = CatBoostClassifier()
cbcf.fit(X_train, y_train,cat_features=cat_features, eval_set=(X_test, y_test),verbose=False)
print(f"Модель обучена: {str(cbcf.is_fitted())}")
cbcf_pred = cbcf.predict(X_test)

cb_matrix = confusion_matrix(y_test, cbcf_pred)
df_cb_matrix = pd.DataFrame(cb_matrix, index=true_names, columns=pred_names)
sns.heatmap(df_cb_matrix, annot=True, fmt='d')
plt.show()

print(classification_report(y_test, cbcf_pred))

cbcf_pred_prob = cbcf.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, cbcf_pred_prob)
roc_auc = auc(fpr, tpr)
print('ROC_AUC:', roc_auc)
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC-curve')
plt.show()
