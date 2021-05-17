import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def Tennis():
    tennis = pd.read_csv('./PlayTennis.csv')
    tennis_original = tennis.copy()

    # Encoding
    le = LabelEncoder()
    for col in tennis.columns:
        tennis[col] = le.fit_transform(tennis[col])

    # x and y split
    y = tennis['Play Tennis']
    x = tennis.drop('Play Tennis', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1)

    # Build three model
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    bnb = BernoulliNB()
    bnb.fit(x_train, y_train)

    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)

    # compare the results
    print('GaussianNB score : ', gnb.score(x_test, y_test))
    print('BernoulliNB score : ', bnb.score(x_test, y_test))
    print('MultinomialNB score : ', mnb.score(x_test, y_test))

    # prediction with test data
    test_data = pd.DataFrame([[1, 0, 1, 1]], columns = x_train.columns)
    print('GaussianNB model predict result : ', gnb.predict(test_data))


if __name__ == '__main__':
    Tennis()
