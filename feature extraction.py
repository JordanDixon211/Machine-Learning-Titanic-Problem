import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv('C:/Users/Jordan/Downloads/train.csv')
data_test = pd.read_csv('C:/Users/Jordan/Downloads/test.csv')

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 9, 15, 27, 62, 600)
    group_names = ['Unknown', 'price1', 'price2', 'price3', 'price4' , 'price5']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df
import seaborn as sns

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



def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 9, 15, 27, 62, 1000)
    group_names = ['Unknown', 'price1', 'price2', 'price3', 'price4' , 'price5']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

data_train = simplify_ages(data_train)
data_train = simplify_fares(data_train)
data_train = simplify_sibling(data_train)


#sns.barplot(x = "Sex" , y = "Survived" , data= data_train)
#sns.barplot(x = "Pclass" , y = "Survived" , hue = "Sex" , data= data_train)
#sns.barplot(x = "Pclass" , y = "Fare"  , data= data_train)
#sns.barplot(x = "Pclass" , y = "Fare"  , data= data_train)
#sns.boxplot(x = "SibSp" , y = "Survived" , data= data_train)
#sns.barplot(x = "Embarked" , y = "Survived" , data= data_train)
#sns.barplot(x = "Survived" , y = "Cabin" , data= data_train)
#sns.barplot(x = "Age" , y = "Survived" ,hue = "Pclass" , data= data_train)
#plt.show()
#sns.barplot(x = "Pclass" , y = "Survived" ,hue = "Sex" , data= data_train)
#sns.barplot(x = "Sex" , y = "Survived"  , hue = "Parch" , data= data_train)
#sns.pairplot(data_train, x_vars=['Fare'] , y_vars='Survived', kind='reg')
sns.barplot(x = "SibSp" , y = "Survived"  ,hue = "Age" ,data= data_train)

plt.show()