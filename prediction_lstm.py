import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

plt.rcParams['figure.figsize'] = (20, 10)

from pandas_datareader.data import DataReader
from datetime import datetime


# INPUT #######                ## Tum proje icin kullanabilegemiz global variablelarimiz.
TECH_LIST = ['TSLA']          ## Buraya istedigimiz tickeri yazabiliriz.
TECH_NAMES = ['Tesla']
MOVING_AVERAGE_DAYS = [10, 20, 30, 50]
END = datetime.now()
START = datetime(END.year - 4, END.month, END.day)
scaler = MinMaxScaler(feature_range=(0,1))


##############################

TICKER_TO_STOCK = {}      ## Bir key istedgimiz zaman suslu parantezle keyin degerini donuyor.Koseli yapsaydik parantez liste olacakti, suslu parantezle dictionary haline getirdik.
TICKER_TO_NAME = {}

def set_globals():
    global TICKER_TO_STOCK
    global TICKER_TO_NAME

    for ticker in TECH_LIST:
        ticker_stock_data = DataReader(ticker, 'yahoo', START, END)
        TICKER_TO_STOCK[ticker] = ticker_stock_data

    for ticker, name in zip (TECH_LIST, TECH_NAMES):
        TICKER_TO_NAME[ticker] = name

    for ticker in TICKER_TO_STOCK:
        stock = TICKER_TO_STOCK[ticker]
        stock["company_name"] = TICKER_TO_NAME[ticker]
        TICKER_TO_STOCK[ticker] = stock


def show_close_price_history(ticker):
    plt.title('Close Price History: ' + TICKER_TO_NAME[ticker])
    plt.plot(TICKER_TO_STOCK[ticker]['Close'])
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price USD ($): ' + TICKER_TO_NAME[ticker], fontsize=14)
    plt.show()


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=60):
    dataX = []
    dataY = []

    for i in range(look_back, len(dataset)):
        dataX.append(dataset[i-look_back:i, 0])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)

def get_train_and_test_data(ticker, look_back): #return train and test data
    stock_data = TICKER_TO_STOCK[ticker].filter(['Close'])
    dataset = stock_data.values

    stock_data_scaled = scaler.fit_transform(dataset)

    train_size = int(np.ceil(len(dataset) * 0.8))
    test_size = len(dataset) - train_size

    train_data = stock_data_scaled[0:train_size,:]
    test_data = stock_data_scaled[train_size - look_back:,:]#anlamadik!!!!

    return train_data, test_data

def reshape_data(ticker, look_back):
    train_data, test_data = get_train_and_test_data(ticker, look_back)
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    return trainX, trainY, len(train_data), testX, testY, len(test_data)

def predict_with_lstm(ticker, look_back):#https://www.kaggle.com/faressayah/stock-market-analysis-prediction-using-lstm
    trainX, trainY, train_len, testX, testY, test_len = reshape_data(ticker, look_back)

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(trainX.shape[1],1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(trainX, trainY, batch_size=1, epochs=1)

    predictions = model.predict(testX)
    predictions = scaler.inverse_transform(predictions)

    root_mean_squared_error = np.sqrt(np.mean(((predictions - testY) ** 2)))

    print('Error is: ' + str(root_mean_squared_error))

    #plot the data
    train = TICKER_TO_STOCK[ticker]['Close'][:train_len]
    expected = TICKER_TO_STOCK[ticker][train_len:]
    expected['Predictions'] = predictions
    plt.title('LSTM Results for ' + TICKER_TO_NAME[ticker])
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price USD ($)', fontsize=14)
    plt.plot(train)
    plt.plot(expected[['Close', 'Predictions']])
    plt.legend(['Train', 'Expected', 'Predictions'], loc='lower right')
    plt.show()






set_globals()

for ticker in TECH_LIST:
    show_close_price_history(ticker)
    predict_with_lstm(ticker, look_back=60)
