import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn import datasets


def Question_1_2():
    df = pd.read_excel('./df1.xlsx', engine='openpyxl')
    # Question 1
    print("Result of Question 1")
    for i in range(len(df.columns)):
        print('The number of nan in ', str(i), 'th column : ', df[df.columns[i]].isna().sum())

    # Question 2
    df.age.fillna(df.age.mean(), inplace=True)

    # check question 2 result
    print("\nResult of Question 2\n", df, '\n')


def Question_3():
    df2 = pd.read_excel('./df2.xlsx', engine='openpyxl')
    # Question 3
    df2.age.replace(25, 26, inplace=True)
    df2.name.replace('john', 'johnny', inplace=True)

    # check question 3 result
    print("\nResult of Question 3\n", df2, '\n')


def Question_4():
    df3 = pd.read_excel('./df3.xlsx', engine='openpyxl')
    # Question 4
    le = LabelEncoder()
    df3.Country = le.fit_transform(df3.Country)
    df3.Purchased = le.fit_transform(df3.Purchased)
    
    # check question 4 result
    print("\nResult of Question 4\n", df3, '\n')


def Question_5_6():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['class'] = iris.target
    
    x = df.drop('class', axis=1)
    y = df['class']

    # Question 5
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)
    # Question 6
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(x_train, y_train)
    CV_score = cross_val_score(LDA, x_test, y_test, scoring='accuracy', cv=5)
    print("cv score : ", CV_score)
    print("cv score mean : ", CV_score.mean())
    print("cv score std : ", CV_score.std())


def Question_7():
    df4 = pd.read_excel('./df4.xlsx', engine='openpyxl')
    # Question 7
    x_train = df4.drop('운전시간', axis=1)
    y_train = df4['운전시간']

    model = LinearRegression()
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_train[:3])
    print("\nr squared score : ", r2_score(y_train[:3], y_pred))
    

def Question_8_9_10():
    bigmart = pd.read_csv('./bigmart_train.csv')
    # Question 8
    sales = bigmart.groupby(by = 'Item_Type').sum()
    sns.barplot(sales.index, sales.Item_Outlet_Sales, palette='pastel')
    plt.xticks(rotation=45)
    plt.title('Item Type VS Sales')
    plt.show()

    # Question 9
    sns.barplot(bigmart.Item_Identifier.value_counts()[:10].index, bigmart.Item_Identifier.value_counts()[:10], palette='pastel')
    plt.xticks(rotation=45)
    plt.title('Count of Identifier')
    plt.show()

    # Question 10
    types = bigmart.groupby(by = 'Item_Type').count()
    sns.barplot(types.Item_Identifier, types.index, palette='pastel')
    plt.title('Count of Item Type')
    plt.show()

if __name__ == '__main__':
    Question_1_2()
    #Question_3()
    #Question_4()
    #Question_5_6()
    #Question_7()
    #Question_8_9_10()
