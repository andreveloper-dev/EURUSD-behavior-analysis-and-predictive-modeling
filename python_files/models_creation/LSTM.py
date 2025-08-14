from os import path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

def load_and_prepare_data(dataset_path):
    global X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, scaler
    pd.set_option('display.max_columns', None)
    dataset = pd.read_excel(dataset_path)
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset.set_index('date', inplace=True)

    X = dataset.drop(['eur_usd'], axis=1)
    y = dataset['eur_usd']

    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]

    features = dataset.drop(['eur_usd'], axis=1).columns.tolist()
    vif_data = pd.DataFrame()
    vif_data["Variable"] = features
    vif_data["VIF"] = [variance_inflation_factor(dataset[features].dropna().values, i) for i in range(len(features))]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = scaler.transform(y_val.values.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

    def create_sequences(features, target, time_steps=10):
        X_seq, y_seq = [], []
        for i in range(len(features) - time_steps):
            v = features[i:(i + time_steps)]
            X_seq.append(v)
            y_seq.append(target[i + time_steps])
        return np.array(X_seq), np.array(y_seq)

    time_steps = 100

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

    build_model()
    model_and_training()


def build_model():
    global model
    model = Sequential()
    model.add(LSTM(units=416,
                   activation='tanh',
                   input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                   return_sequences=False))
    model.add(Dropout(0.0))
    model.add(Dense(units=224, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mape'])


def model_and_training():
    model_path = 'lstm_model.h5'
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train_seq, y_train_seq,
                        epochs=200,
                        validation_data=(X_val_seq, y_val_seq),
                        callbacks=[early_stopping])
    model.save(model_path)
data_path = ''
load_and_prepare_data(data_path)