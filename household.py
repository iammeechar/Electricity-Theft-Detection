import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.preprocessing
import seaborn as sns
from numpy import nan
from numpy import isnan
from pandas import read_csv
from numpy import split
from numpy import array
from matplotlib import pyplot

#load the data
dataset = read_csv('household_power_consumption.txt', sep = ';', header = 0, low_memory = False, infer_datetime_format = True, parse_dates = {'datetime': [0,1]},index_col = ['datetime'])

#mark all missing values
dataset.replace('?', nan, inplace = True)
#make dataset numeric
dataset = dataset.astype('float32')

#filling missing values with a value at the same time as the one the previous day
def fill_missing(values):
    one_day = 60 * 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
                        if isnan(values[row, col]):
                            values[row, col] = values[row - one_day, col]
                        
#fill missing
fill_missing(dataset.values)

#add a column for the remainder of sub metereing
values = dataset.values
dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5]
                                                        + values[:,6])

#save updated dataset
dataset.to_csv('household_power_consumption.csv')

#load the new file
dataset = read_csv('household_power_consumption.csv', header = 0, infer_datetime_format = True, parse_dates = ['datetime'], index_col = ['datetime'])
print(dataset.head())

#line plot for each variable
pyplot.figure()
for i in range(len(dataset.columns)):
    plt.figure(figsize = (16,6))
    #create subplot
    pyplot.subplot(len(dataset.columns), 1, i+1)
    #get variable name
    name = dataset.columns[i]
    #plot data
    pyplot.plot(dataset[name])
    #set title
    pyplot.title(name, y=0)
    #turn of ticks to remove clutter
    pyplot.yticks([])
    pyplot.xticks([])
pyplot.show()

#yearly line plots for power usage dataset
#plot active poer for each year
years = ['2007', '2008', '2009', '2010']
pyplot.figure()
for i in range(len(years)):
    plt.figure(figsize = (16,6))
    #prepare subplot
    ax = pyplot.subplot(len(years), 1, i+1)
    #determine the year to plot
    year = years[i]
    #get all observations for the year
    result = dataset[str(year)]
    #plot the active power for the year
    pyplot.plot(result['Global_active_power'])
    #add title to the subplot
    pyplot.title(str(year), y=0, loc ='left')
    #turn of ticks to remove clutter
    pyplot.yticks([])
    pyplot.xticks([])
pyplot.show()

#monthly line plots for power usage dataset
#plot active power for each year
months = [x for x in range(1, 13)]
pyplot.figure()
for i in range(len(months)):
    plt.figure(figsize = (16,6))
    #prepare subplot
    ax = pyplot.subplot(len(months), 1, i+1)
    #determine the month to plot
    month = '2007-' + str(months[i])
    #get all observations for the year
    result = dataset[month]
    #plot the active power for the year
    pyplot.plot(result['Global_active_power'])
    #add title to the subplot
    pyplot.title(month, y=0, loc ='left')
    #turn of ticks to remove clutter
    pyplot.yticks([])
    pyplot.xticks([])
pyplot.show()

pyplot.figure()
for i in range(len(months)):
    plt.figure(figsize = (16,6))
    #prepare subplot
    ax = pyplot.subplot(len(months), 1, i+1)
    #determine the month to plot
    month = '2008-' + str(months[i])
    #get all observations for the year
    result = dataset[month]
    #plot the active power for the year
    pyplot.plot(result['Global_active_power'])
    #add title to the subplot
    pyplot.title(month, y=0, loc ='left')
    #turn of ticks to remove clutter
    pyplot.yticks([])
    pyplot.xticks([])
pyplot.show()

pyplot.figure()
for i in range(len(months)):
    plt.figure(figsize = (16,6))
    #prepare subplot
    ax = pyplot.subplot(len(months), 1, i+1)
    #determine the month to plot
    month = '2009-' + str(months[i])
    #get all observations for the year
    result = dataset[month]
    #plot the active power for the year
    pyplot.plot(result['Global_active_power'])
    #add title to the subplot
    pyplot.title(month, y=0, loc ='left')
    #turn of ticks to remove clutter
    pyplot.yticks([])
    pyplot.xticks([])
pyplot.show()


#daily line plots for power usage dataset
days = [x for x in range(1, 20)]
pyplot.figure()
for i in range(len(days)):
    plt.figure(figsize = (16,6))
    #prepare subplot
    ax = pyplot.subplot(len(days), 1, i+1)
    #determine the month to plot
    day = '2007-01-' + str(days[i])
    #get all observations for the year
    result = dataset[day]
    #plot the active power for the year
    pyplot.plot(result['Global_active_power'])
    #add title to the subplot
    pyplot.title(day, y=0, loc ='left')
    #turn of ticks to remove clutter
    pyplot.yticks([])
    pyplot.xticks([])
pyplot.show()



#histogram plots for power usage dataset
#histogram plot for each dataset
pyplot.figure()
for i in range(len(dataset.columns)):
    plt.figure(figsize = (16,6))
    #create subplot
    pyplot.subplot(len(dataset.columns), 1, i+1)
    #get variable name 
    name = dataset.columns[i]
    #create histogram
    dataset[name].hist(bins=100)
    #add title to the subplot
    pyplot.title(name, y=0, loc ='left')
    #turn of ticks to remove clutter
    pyplot.yticks([])
    pyplot.xticks([])
pyplot.show()

#resample data to daily
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()

#summarize()
print(daily_data.shape)
print(daily_data.head())

#save
daily_data.to_csv('household_power_consumption_days.csv')

#evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    #calculate the RMSE score for each day
    for i in range(actual.shape[1]):
        #calculate mse
        mse = mean_square_error(actual[:, i], predicted[:,i])
        #calculate rmse
        rmse = sqrt(mse)
        #store
        scores.append(rmse)
    #calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) **2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores
#split a univariate dataset into train/test sets
#scaling numeric columns 
scaler = sklearn.preprocessing.MinMaxScaler()
dataset['Global_reactive_power'] = scaler.fit_transform(dataset['Global_reactive_power'].values.reshape(-1,1))
dataset['Voltage'] = scaler.fit_transform(dataset['Voltage'].values.reshape(-1,1))
dataset['Global_intensity'] = scaler.fit_transform(dataset['Global_intensity'].values.reshape(-1,1))
dataset['Sub_metering_1'] = scaler.fit_transform(dataset['Sub_metering_1'].values.reshape(-1,1))
dataset['Sub_metering_2'] = scaler.fit_transform(dataset['Sub_metering_2'].values.reshape(-1,1))
dataset['Sub_metering_3'] = scaler.fit_transform(dataset['Sub_metering_3'].values.reshape(-1,1))
dataset['Global_active_power'] = scaler.fit_transform(dataset['Global_active_power'].values.reshape(-1,1))

#use 90% for training
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset.iloc[0:train_size], dataset.iloc[train_size:len(dataset)]

#create tensors for inputting data into the network
def create_dataset(X, y, time_steps = 1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
#with a sequence length of 10, create training and testing datasets
time_steps = 10
X_train, y_train = create_dataset(train, train.Global_active_power, time_steps)
X_test, y_test = create_dataset(test, test.Global_active_power, time_steps)

#slice datasets and create batches for an improved training
batch_size = 256
buffer_size = 1000

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()

test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_data = test_data.batch(batch_size).repeat()

#creating a CNN Model
simple_lstm_model = tf.keras.models.Sequential([tf.keras.layers.LSTM
                                               (8, input_shape = X_train.shape[-2:])
                                               ,tf.keras.layers.Dense(1)])
simple_lstm_model.compile(optimizer = 'adam', loss = 'mae')

#training
EVALUATION_INTERVAL = 200
EPOCHS = 10

history = simple_lstm_model.fit(train_data, epochs = EPOCHS,
                               steps_per_epoch = EVALUATION_INTERVAL,
                               validation_data = test_data,
                               validation_steps = 50)

#after training is finished, plot losses
#plot losses
plt.plot(history.history['loss'], label = 'train loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


X_test, y_test = create_dataset(dataset, dataset.Global_active_power, 10)
y_pred = simple_lstm_model.predict(X_test)

#plot prediction to get some visualization
def create_time_steps(length):
    return list(range(-length, 0))

#plotting code 
plt.figure(figsize = (16, 4))
num_in = create_time_steps(1644626)
num_out = 28
#plt.plot(num_in, y_train[15571:], label = 'history', color="green")
plt.plot(np.arange(num_out), y_test[15661:15689], 'b', label = 'Actual', color="red")
plt.plot(np.arange(num_out), y_pred[15661:15689], 'b', label = 'Predicted')
plt.xlabel("Time")
plt.ylabel("Global active power(Normalized Value)")
plt.legend()

plt.show()

#predict the future point
y_pred = simple_lstm_model.predict(X_test[-1:])
#print value
y_pred
#plot prediction 
plt.figure(figsize = (16, 4))
num_in = create_time_steps(100)
num_out = 1
plt.plot(num_in, y_test[-100:])
plt.plot(np.arange(num_out), y_pred, 'ro', label ='Predicted')
plt.xlabel("Time")
plt.ylabel("Active Global power (Normalized Value)")
plt.legend()
plt.show()

#predicting range of data points
dataset2 = dataset['2010-01-01 00:00:00' : '2010-11-26 21:02:00']
dataset1 = dataset['2009-01-01 00:00:00' : '2009-12-31 21:02:00']
dataset1['Global_active_power'] = 0
dataset_future = dataset2.append(dataset1, sort = False)
dataset_future
#for cleanliness and avoiding dicontinuity use the following code 
#df_future = df_future.reset_index(drop = True)
#perform predictions in a continous loop adding the predicted value to the df_future dataframe
predictions = []
#predicton loop
for i in range(50):
    X_f, y_f = create_dataset(dataset_future, dataset_future.Global_active_power, time_steps)
    y_pred = simple_lstm_model.predict(X_f[i:i+1])
    dataset_future['Global_active_power'][i+10] = y_pred
    predictions.append(float(y_pred[0][0]))
#plot predictions
plt.figure(figsize = (16, 4))
num_in = create_time_steps(100)
num_out = 50
plt.plot(num_in, y_test[-100:], label = 'History')
plt.plot(np.arange(num_out), predictions, 'r', label ='Predicted')
plt.xlabel("Time")
plt.ylabel("Global active power(Normalized Value)")
plt.legend()
plt.show()