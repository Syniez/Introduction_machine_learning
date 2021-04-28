from IPython.display import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydotplus

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def Iris():
    # Load iris dataset
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df['class'] = iris.target

    # Separate x and y data from DataFrame
    x = df.drop('class', axis=1)
    y = df['class']

    # Use 80% for training, 20% for testing 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)

    # Decision tree
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(x_train, y_train)

    # Decision tree model evaluation
    dt_pred = dt.predict(x_test)
    dt_score = accuracy_score(y_test, dt_pred)
    print("\nDecision tree prediction score :", dt_score)

    # 5 fold cross validation
    dt_cv = cross_val_score(dt, x_train, y_train, cv=5, scoring='accuracy')
    print("Cross validation score :", dt_cv),
    print("Mean score :", dt_cv.mean())
    print("Standard deviation score :", dt_cv.std())

    # Decision tree visualization and save as .png
    dot_data = tree.export_graphviz(dt, feature_names=iris.feature_names, class_names=iris.target_names)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("iris.png")

    # Random forest
    print("\n-----------------------------\n")
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    # Random forest model evaluation
    rf_pred = rf.predict(x_test)
    rf_score = accuracy_score(y_test, rf_pred)
    print("Random forest prediction score :", rf_score)

    # 5 fold cross validation
    rf_cv = cross_val_score(rf, x_train, y_train, cv=5, scoring='accuracy')
    print("Cross validation score :", rf_cv)
    print("Mean score :", rf_cv.mean())
    print("Standard deviation score :", rf_cv.std(), "\n")

    # model comparaision
    means = [dt_cv.mean(), rf_cv.mean()]
    stds = [dt_cv.std(), rf_cv.std()]

    df1 = pd.DataFrame({'mean' : means, 'std' : stds}, index = ['DT', 'RF'])
    df2 = pd.DataFrame({'DT' : dt_cv, 'RF' : rf_cv})

    df2.boxplot()
    plt.show()


def Diabetes():
    # load data and replace number class to name
    data = pd.read_csv('/home/jong/Downloads/pima-indians-diabetes.csv', names=['pregnant', 'plasma', 'presure', 'thickness', 'insulin', 'BMI', 'pedigree', 'age', 'class'])
    data['class'].replace(0, 'Normal', inplace=True)
    data['class'].replace(1, 'Diabetes', inplace=True)

    # Separate x and y data
    x = data.drop('class', axis=1)
    y = data['class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1)

    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(x_train, y_train)

    # decision tree model evaluation
    dt_pred = dt.predict(x_test)
    dt_score = accuracy_score(y_test, dt_pred)
    print(dt_score)

    # cross validation
    dt_cv = cross_val_score(dt, x_train, y_train, cv=5, scoring='accuracy')
    print(dt_cv)
    print(dt_cv.mean())
    print(dt_cv.std())

    # decision tree visualization
    dot_data = tree.export_graphviz(dt, feature_names = data.columns[:8], class_names = ['Normal', 'Diabetes'])
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('pima_indians.png')

    # random forest
    print("\n-----------------------------\n")
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    # random forest model evaluation
    rf_pred = rf.predict(x_test)
    rf_score = accuracy_score(y_test, rf_pred)
    print(rf_score)

    # 5 fold cross validation
    rf_cv = cross_val_score(rf, x_train, y_train, cv=5, scoring='accuracy')
    print("Cross validation score :", rf_cv)
    print("Mean score :", rf_cv.mean())
    print("Standard deviation score :", rf_cv.std(), "\n")

    # model comparaision
    means = [dt_cv.mean(), rf_cv.mean()]
    stds = [dt_cv.std(), rf_cv.std()]

    df1 = pd.DataFrame({'mean' : means, 'std' : stds}, index = ['DT', 'RF'])
    df2 = pd.DataFrame({'DT' : dt_cv, 'RF' : rf_cv})

    df2.boxplot()
    plt.show()


if __name__ == '__main__':
    Iris()
    #Diabetes()