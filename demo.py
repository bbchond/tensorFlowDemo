import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import tensorflow.keras
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import datetime

from sklearn import preprocessing

if __name__ == '__main__':
    features = pd.read_csv("temps.csv")
    print('数据维度:', features.shape)

    years = features['year']
    months = features['month']
    days = features['day']

    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

    plt.style.use('fivethirtyeight')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.autofmt_xdate(rotation=45)

    # 标签值
    ax1.plot(dates, features['actual'])
    ax1.set_xlabel('')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Max Temp')

    # 昨天
    ax2.plot(dates, features['temp_1'])
    ax2.set_xlabel('')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Previous Max Temp')

    # 前天
    ax3.plot(dates, features['temp_2'])
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Two Days Prior Max Temp')

    # 我的逗逼朋友
    ax4.plot(dates, features['friend'])
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Temperature')
    ax4.set_title('Friend Estimate')

    plt.tight_layout(pad=2)

    features = pd.get_dummies(features)

    features = pd.get_dummies(features)
    features.head(5)

    labels = np.array(features['actual'])
    features = features.drop('actual', axis=1)

    feature_list = list(features.columns)

    features = np.array(features)
    input_features = preprocessing.StandardScaler().fit_transform(features)

    model = tf.keras.Sequential()
    model.add(layers.Dense(16, kernel_initializer='random_normal'))
    model.add(layers.Dense(32, kernel_initializer='random_normal'))
    model.add(layers.Dense(1, kernel_initializer='random_normal' ))
    # 对网络进行配置，指定好优化器和损失函数等，此处指定mse损失函数
    model.compile(optimizer=tf.keras.optimizers.SGD(0.001),
                  loss='mean_squared_error')
    model.fit(input_features, labels, validation_split=0.25, epochs=10, batch_size=64)
    print(model)

    model.summary()
