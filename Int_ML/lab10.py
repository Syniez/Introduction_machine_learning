import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_roc_curve


def College():
    # Load .csv file data from same directory 
    college = pd.read_csv('./College.csv')
    college.set_index('Unnamed: 0', inplace = True)
    
    y = college.Private
    x = college.drop('Private', axis=1)

    # Label encoding
    y_orig = y.copy()
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Data plotting
    x.boxplot()
    plt.show()
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    plt.boxplot(x)
    plt.show()

    # Build MLP classifier and train & scoring
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), random_state=0)
    mlp.fit(x_train, y_train)
    print('MLP score : ', mlp.score(x_test, y_test))

    # Model evaluation
    y_pred = mlp.predict(x_test)
    print('MLP accuracy score : ', accuracy_score(y_test, y_pred))
    print('\nConfusion matrix \n', confusion_matrix(y_test, y_pred))
    print('\nClassification report \n', classification_report(y_test, y_pred))
    print('\nCross validation score \n', cross_val_score(mlp, x_test, y_test, cv=10, scoring='accuracy'))
    plot_roc_curve(mlp, x_test, y_test)
    plt.show()


if __name__ == '__main__':
    College()
