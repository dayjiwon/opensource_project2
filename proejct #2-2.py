import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def sort_dataset(dataset_df):
    # year값으로 오름차순 정렬
    sorted_df = dataset_df.sort_values(by='year')
    return sorted_df

def split_dataset(dataset_df):
    # salary에 0.001을 곱해 label 값을 rescale해줌
    labels = dataset_df['salary'] * 0.001
    # x는 salary를 제외한 데이터를, y는 label데이터를 저장
    X = dataset_df.drop('salary', axis=1)
    Y = labels
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1718) # [:1718]까지 train으로 설정
    return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
    numerical_cols = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    numerical_df = dataset_df[numerical_cols] # numerical 값들만 추출
    return numerical_df

# DecisionTreeRegressor를 사용하여 train 해주고, 예측 결과 return
def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_rgs = DecisionTreeRegressor()
    dt_rgs.fit(X_train, Y_train)
    dt_predict = dt_rgs.predict(X_test)
    return dt_predict

# RandomForestRegressor를 사용하여 train 해주고, 예측 결과 return
def train_predict_random_forest(X_train, Y_train, X_test):
    rf_rgs = RandomForestRegressor()
    rf_rgs.fit(X_train, Y_train)
    rf_predict = rf_rgs.predict(X_test)
    return rf_predict

# SVR을 사용하여 train 해주고, 예측 결과 return
def train_predict_svm(X_train, Y_train, X_test):
    #pipeline을 사용하여 StandardScaler와 SVR 연결
    svm_pip = make_pipeline(
        StandardScaler(),
        SVR()
    )
    svm_pip.fit(X_train, Y_train)
    svm_predict = svm_pip.predict(X_test)
    return svm_predict

def calculate_RMSE(labels, predictions): # rmse 계산 return 
    RMSE = np.sqrt(np.mean((predictions - labels)**2))
    return RMSE

if __name__=='__main__':
    #DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
