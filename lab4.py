import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_roc_curve

def Loan():
    # load loan_train dataset
    loan_train = pd.read_csv('./data/loan_train.csv')
    
    # 기본탐색
    print(loan_train.head())
    print(loan_train.shape)
    print('\n', loan_train.dtypes)
    columns = loan_train.columns
    for col in columns:
        print(loan_train[col].value_counts(), '\n')

    # load loan_test dataset
    loan_test = pd.read_csv('./data/loan_test.csv')
    loan = pd.concat([loan_train, loan_test])
    
    # Preprocessing
    loan.LoanAmount.fillna(loan.LoanAmount.mean(), inplace=True)
    names = ['Self_Employed', 'Gender', 'Married', 'Dependents', 'Loan_Amount_Term', 'Credit_History']
    for n in names:
        loan[n].fillna(loan[n].mode()[0], inplace=True)

    # Use label encoder for Gender, Married, Education, Self_Employed
    loan_original = loan.copy()
    bin_names = ['Gender', 'Married', 'Education', 'Self_Employed']
    le = LabelEncoder()
    for b in bin_names:
        loan[b] = le.fit_transform(loan[b])

    loan.Loan_Status.replace({'Y':1, 'N':0}, inplace=True)
    loan.Dependents.replace('3+', '3', inplace=True)
    loan.Dependents = loan.Dependents.astype('int32')

    # train dand test data split
    data = loan[:614]
    test_data = loan[614:]

    # x and y split (80% for training, 20% for testing)
    y = data.Loan_Status
    x = data.drop(['Loan_ID', 'Loan_Status', 'Property_Area'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    
    # Build a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # Evaluation
    print('Model score :', model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    print('Accuracy score :', accuracy_score(y_test, y_pred), '\n')

    # 10-fold cross validation
    cv = cross_val_score(model, x_test, y_test, cv=10, scoring='accuracy')
    print('10-fold cross validation mean :', cv.mean())
    print('10-fold cross validation std :', cv.std(), '\n')

    # Confusion matrix, classification report, ROC curve
    print('Confusion matrix\n', confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_roc_curve(model, x_test, y_test)
    plt.show()

    # Predict with test data
    test_data = test_data.drop(['Loan_ID', 'Property_Area', 'Loan_Status'], axis=1)
    pred_result = model.predict(test_data)
    print('Model prediction result :\n', pred_result)


if __name__ == '__main__':
    Loan()