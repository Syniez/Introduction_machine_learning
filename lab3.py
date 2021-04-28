import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets


def Iris():
    # load iris dataset and make DataFrame
    Iris = datasets.load_iris()
    df = pd.DataFrame(Iris.data)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df['species'] = Iris.target

    # x and y split
    y = df.species
    x = df.drop('species', axis=1)
    # train and test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)

    # load and save 5 different models
    models = [DecisionTreeClassifier(), RandomForestClassifier(), LinearDiscriminantAnalysis(), KNeighborsClassifier(), SVC()]
    scores = []
    cvs = []

    # evaluation of each five models
    for model in models:
        model.fit(x_train, y_train)
        scores.append(model.score(x_test, y_test))
        y_pred = model.predict(x_test)
        cvs.append(cross_val_score(model, x_test, y_test, scoring='accuracy', cv=6))

    # compare five different models performance
    df_cvs = pd.DataFrame(cvs).T
    df_cvs.columns = ['DT', 'RF', 'LDA', 'KNN', 'SVM']
    df_cvs.boxplot()
    plt.show()

    summary_df = pd.concat([df_cvs.mean(), df_cvs.std()], axis=1)
    summary_df.columns = ['mean', 'std']
    scores_df = pd.DataFrame(scores).T
    scores_df.columns = ['DT', 'RF', 'LDA', 'KNN', 'SVM']
    print(scores_df, '\n')
    print(summary_df)

    # load best model(LDA) and predict with test data
    test_data = pd.DataFrame({'sepal_length':6.2, 'sepal_width':3.1, 'petal_length':4.3, 'petal_width':1.1}, index=[0])
    models[2].fit(x_train, y_train)
    lda_pred = models[2].predict(test_data)


def Diabetes():
    # load data and replace number class to name
    data = pd.read_csv('./pima-indians-diabetes.csv', names=['pregnant', 'plasma', 'presure', 'thickness', 'insulin', 'BMI', 'pedigree', 'age', 'class'])
    data['class'].replace(0, 'Normal', inplace=True)
    data['class'].replace(1, 'Diabetes', inplace=True)

    # x and y split
    x = data.drop('class', axis=1)
    y = data['class']
    # train and test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)

    # load and save 5 different models
    models = [DecisionTreeClassifier(), RandomForestClassifier(), LinearDiscriminantAnalysis(), KNeighborsClassifier(), SVC()]
    scores = []
    cvs = []

    # evaluation of each five models
    for model in models:
        model.fit(x_train, y_train)
        scores.append(model.score(x_test, y_test))
        y_pred = model.predict(x_test)
        cvs.append(cross_val_score(model, x_test, y_test, scoring='accuracy', cv=10))

    # compare five different models performance
    df_cvs = pd.DataFrame(cvs).T
    df_cvs.columns = ['DT', 'RF', 'LDA', 'KNN', 'SVM']
    df_cvs.boxplot()
    plt.show()

    summary_df = pd.concat([df_cvs.mean(), df_cvs.std()], axis=1)
    summary_df.columns = ['mean', 'std']
    scores_df = pd.DataFrame(scores).T
    scores_df.columns = ['DT', 'RF', 'LDA', 'KNN', 'SVM']
    print(scores_df, '\n')
    print(summary_df)

    # load best model(LDA) and predict with test data
    test_data = pd.DataFrame({'pregnant':5.1, 'plasma':1.3, 'presure':2.7, 'thickness':3.5, 'insulin':1.1, 'BMI':7.2, 'pedigree':1.5, 'age':4.3}, index=[0])
    models[2].fit(x_train, y_train)
    lda_pred = models[2].predict(test_data)


if __name__ == '__main__':
    Iris()
    #Diabetes()