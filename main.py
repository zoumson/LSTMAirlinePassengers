import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(2)


# Forecast Time Series with LSTM

# convert an array of values into a dataset matrix
def create_dataset(dataset_fct, look_back_fct=1):
    datax, datay = [], []
    for i in range(len(dataset_fct) - look_back_fct - 1):
        a = dataset_fct[i:(i + look_back_fct), 0]
        datax.append(a)
        datay.append(dataset_fct[i + look_back_fct, 0])
    return numpy.array(datax), numpy.array(datay)


if __name__ == '__main__':

    # load the dataset
    # use the second column passenger, that is usecols=[1]
    dataframe = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')

    # get the column passenger in a numpy array form
    # this removes the colum name aslo
    dataset = dataframe.values

    # set the data type from int64 to float32
    dataset = dataset.astype('float32')

    # min max normalization the dataset in the range of 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))

    # apply the normalization to the dataset
    dataset = scaler.fit_transform(dataset)

    print(dataset.shape)

    # split into train 67% and test 33% sets
    train_size = int(len(dataset) * 0.67)

    test_size = len(dataset) - train_size
    # dataset size is (144, 1)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    # 1 time ahead forcast, use current time as a feature, its label is the next time data
    # sequence is days
    # timestamp is the sequence of the days
    # t-1 is used to predict the number of passenger at t, so timestamp is set to  1
    look_back = 5
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset), label="original data")
    plt.plot(trainPredictPlot, label="train data")
    plt.plot(testPredictPlot, label="test data")
    plt.legend(loc="upper left")
    plt.savefig('passenger.png')
    plt.show()
