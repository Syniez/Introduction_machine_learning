import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np


def Iris():
    # load data and make dataframe
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # replace numeric category to string
    df['class'] = iris.target
    df['class'].replace(0, 'setosa', inplace=True)
    df['class'].replace(1, 'versicolor', inplace=True)
    df['class'].replace(2, 'virginica', inplace=True)

    # summarize dataframe
    print("\nDataFrame shape :",df.shape, "\n")
    print(df.head(20), "\n")
    print(df.describe(), "\n")
    print(df.groupby('class').size(), "\n")
    print(df.dtypes)

    # visualize dataframe
    tab = df.groupby('class').size()
    pct = tab / tab.sum() * 100
    tab = pd.concat([tab, pct], axis=1)
    tab.columns = ['freq', 'percentage']

    _, pos = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    pos[0].bar(tab.index, tab.freq)
    pos[0].set_title("Frequency")
    pos[1].bar(tab.index, tab.percentage)
    pos[1].set_title("Percentage (%)")
    plt.show()

    df.boxplot()
    plt.title("Box plot")
    plt.show()

    pd.plotting.scatter_matrix(df)
    plt.show()


def Diabetes():
    # load data from .csv file
    data = pd.read_csv('/home/jong/Downloads/pima-indians-diabetes.csv', names=['pregnant', 'plasma', 'presure', 'thickness', 'insulin', 'BMI', 'pedigree', 'age', 'class'])

    # replace number class to name
    data['class'].replace(0, 'Normal', inplace=True)
    data['class'].replace(1, 'Diabetes', inplace=True)

    # summarize dataframe
    print("\nDataFrame shape :",data.shape, "\n")
    print(data.head(20), "\n")
    print(data.describe(), "\n")
    print(data.groupby('class').size(), "\n")
    print(data.dtypes)

    # visualize dataframe
    tab = data.groupby('class').size()
    pct = tab / tab.sum() * 100
    tab = pd.concat([tab, pct], axis=1)
    tab.columns = ['freq', 'percentage']

    _, pos = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    pos[0].bar(tab.index, tab.freq)
    pos[0].set_title("Frequency")
    pos[1].bar(tab.index, tab.percentage)
    pos[1].set_title("Percentage (%)")
    plt.show()
    
    data.boxplot()
    plt.title("Box plot")
    plt.show()

    pd.plotting.scatter_matrix(data)
    plt.show()


if __name__ == '__main__':
    Iris()
    #Diabetes()