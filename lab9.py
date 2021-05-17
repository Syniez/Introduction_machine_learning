import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Binarizer


def Diabetes():
    pima = pd.read_csv('./pima-indians-diabetes.csv', header=None)
    pima.columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

    y = pima['class']
    x = pima.drop('class', axis=1)

    # MinMax Scaler -> divide by max value
    scaler = MinMaxScaler()
    x1 = scaler.fit_transform(x)
    plt.boxplot(x1)
    plt.show()

    # Standard Scaler -> make mean=0, sigma=1
    scaler2 = StandardScaler()
    x2 = scaler2.fit_transform(x)
    plt.boxplot(x2)
    plt.show()

    # Binarizer -> make value 0 or 1 (thresholding)
    scaler3 = Binarizer()
    x3 = scaler3.transform(x)
    plt.boxplot(x3)
    plt.show()


if __name__ == '__main__':
    print('hello')
    Diabetes()