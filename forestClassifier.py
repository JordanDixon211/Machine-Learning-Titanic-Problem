import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#simplify the data, and transform some parts.
#[- 1, 0 , 9(lowest value of pclass 3) , 15 (lower mean of pclass 2) ,  27 (upper mean of pclass 2) , 62 (average of pclass 3) , 513 anomaly ]

#Survived	Pclass	Sex	groupnamesAge	SibSp	Parch		Fare		Embarked
#0	3	male	Young Adult	0	0		Price1		Q

data_train = pd.read_csv('C:/Users/Jordan/Downloads/train.csv')
data_test = pd.read_csv('C:/Users/Jordan/Downloads/test.csv')

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 9, 15, 27, 62, 600)
    group_names = ['Unknown', 'price1', 'price2', 'price3', 'price4' , 'price5']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 3, 13, 18, 25, 35, 60, 100)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_sibling(df):
    bins = (-1, 0, 1, 2, 3, 4, 8)
    group_names = ['Zero', 'one', 'two', 'three', 'four' , 'eight']
    categories = pd.cut(df.SibSp, bins, labels=group_names)
    df.SibSp = categories
    return df


def drop_features(df):
    return df.drop(['Ticket', 'Name' , 'Embarked', 'Cabin' ], axis=1)


def transform_features(df):
    df = simplify_ages(df)
    df = simplify_fares(df)
    df = simplify_sibling(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)

#encoding of the data transform the string values into representable numbers such as 0 and 1
def encode_features(df_train, df_test):
    features = ['Fare', 'Age', 'Sex', 'SibSp']
    df_combined = pd.concat([df_train[features], df_test[features]])
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

data_train, data_test = encode_features(data_train, data_test)


X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

X_all_test = data_test.drop(['Survived', 'PassengerId'], axis=1)
y_all_test = data_test['Survived']

#best combo
clf = RandomForestClassifier(n_estimators=18, max_depth=2)
clf.fit(X_all, y_all)
score = 100 * round(clf.score(X_all_test , y_all_test) , 2)
print(score)

num_test = 0.5
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

#best combo
clf = RandomForestClassifier(n_estimators=85, max_depth=10)
clf.fit(X_train, y_train)
y_predictTree = clf.predict(X_test)
score = metrics.accuracy_score(y_predictTree, y_test)
print(score)


t_range = range(1,200)
scores = []
for t in t_range:
    clf = RandomForestClassifier(t)
    clf.fit(X_all, y_all)
    scores.append(100 * clf.score(X_all_test , y_all_test )) ;


plt.plot(t_range, scores)
plt.xlabel("amount of subtrees ")
plt.ylabel("test accuracy")
plt.show()




t_range = range(1,200)
scores = []
for t in t_range:
    clf = RandomForestClassifier(t)
    clf.fit(X_train, y_train)
    y_predictTree = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_predictTree, y_test))

plt.plot(t_range, scores)
plt.xlabel("amount of subtrees ")
plt.ylabel("test accuracy")
plt.show()
