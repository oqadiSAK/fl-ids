import pandas as pd
from sklearn import preprocessing

def get_data():
    train_data = _load_data("data/UNSW_NB15_training-set.csv")
    test_data = _load_data("data/UNSW_NB15_testing-set.csv")

    train_data = _preprocess_data(train_data)
    test_data = _preprocess_data(test_data)

    X_train, Y_train = _separate_features_and_labels(train_data)
    X_test, Y_test = _separate_features_and_labels(test_data)

    return X_train, Y_train, X_test, Y_test

def _load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=["id", "attack_cat"])
    return data

def _preprocess_data(data):
    for column in data.columns:
        if data[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            data[column] = le.fit_transform(data[column])

    min_max_scaler = preprocessing.MinMaxScaler()
    for column in data.columns:
        data[column] = min_max_scaler.fit_transform(data[column].values.reshape(-1,1))
    return data

def _separate_features_and_labels(data):
    Y = data.label.to_numpy()
    X = data.drop(columns="label").to_numpy()
    return X, Y