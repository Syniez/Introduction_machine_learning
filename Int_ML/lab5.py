import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

def Bigmart():
    # load train and test data
    train = pd.read_csv('./bigmart_train.csv')
    test = pd.read_csv('./bigmart_test.csv')

    # Train data 기본탐색
    print('Train data shape : ', train.shape, '\n')
    print(train.dtypes, '\n')
    print(train.describe(include='all'), '\n')
    print(train.isna().sum(), '\n')

    # Test data Item_Outlet_Sales열 생성
    print('Test data shape : ', test.shape, '\n')
    test['Item_Outlet_Sales'] = 1

    # DataFrame 생성 및 전처리
    df = pd.concat([train, test])
    
    # Replacement, fill, drop
    df.Item_Weight.fillna(df.Item_Weight.median(), inplace=True)
    df.loc[df.Item_Visibility==0, 'Item_Visibility'] = df.Item_Visibility.median()
    df.Outlet_Size.value_counts(dropna=False)
    df.Outlet_Size.fillna('Other', inplace=True)
    df.Item_Fat_Content.replace(['LF', 'Low fat'], 'Low fat', inplace=True)
    df.Item_Fat_Content.replace('reg', 'Regular', inplace=True)
    df['Year'] = 2021 - df.Outlet_Establishment_Year
    df = df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Fat_Content', 'Outlet_Establishment_Year', 'Item_Type'], axis=1)   

    # Encoding some data
    le = LabelEncoder()
    df.Outlet_Size = le.fit_transform(df.Outlet_Size)
    df.Outlet_Location_Type = le.fit_transform(df.Outlet_Location_Type)
    df.Outlet_Type = le.fit_transform(df.Outlet_Type)

    # Split train data and test data from df
    train_data = df[:8523]
    test_data = df[8523:]

    # x and y split
    y = train_data.Item_Outlet_Sales
    x = train_data.drop('Item_Outlet_Sales', axis=1)

    # Split train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1)
    
    # Regression
    model = LinearRegression()
    model.fit(x_train, y_train)
    print('Regression score : ', model.score(x_test, y_test))

    # Compute r2 score
    y_pred = model.predict(x_test)
    print('r2 score : ', r2_score(y_test, y_pred))

    # Compute residual before plot residplot
    residual = y_test - y_pred
    std_residual = residual / np.std(residual)

    

if __name__ == '__main__':
    Bigmart()
