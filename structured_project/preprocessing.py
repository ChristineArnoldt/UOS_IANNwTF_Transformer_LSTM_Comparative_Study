import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np



def stock_data(seq_length):
    df = pd.read_csv("IBM.csv", index_col='Date', parse_dates=["Date"])

    training_set = df[:'2016'].iloc[:, 1:2].values
    test_set = df['2017':].iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(seq_length, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - seq_length:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    inputs = df["High"][len(df["High"]) - len(test_set) - seq_length:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(seq_length, len(inputs)):
        X_test.append(inputs[i - seq_length:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return sc, X_train, y_train, X_test, test_set
