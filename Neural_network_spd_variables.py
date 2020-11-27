import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
#Extracting data into a csv file
vehicle_data = pd.read_csv("Z:\\Vehicle_spd_Estimation\\Data_13_07_2020\\Homogenous.csv")#U:\Vehicle_spd_Estimation\Dataset
print(vehicle_data.columns.tolist())

fin_vehicle_data = vehicle_data.drop(['VxVeh_kph' ],axis = 1)
for i in range(0,11):
    fin_vehicle_data.drop(fin_vehicle_data.index[0],inplace = True)
#print (fin_vehicle_data)
plt.scatter(fin_vehicle_data['VxWhl1'],fin_vehicle_data['VxVeh_ms'])
plt.show()
Wheel_speed_feature = fin_vehicle_data.drop(['VxVeh_ms'],axis = 1)
vehicle_speed_label = fin_vehicle_data['VxVeh_ms']
whlspd_train, whlspd_test, VxVeh_train, VxVeh_test = train_test_split(Wheel_speed_feature, vehicle_speed_label, test_size = 0.2, random_state = 0)
#print (whlspd_train.head())
print (VxVeh_train.head())

#train and test data.
whlspd_train_ntime = whlspd_train.drop(['Time    ' ],axis = 1)
whlspd_test_ntime = whlspd_test.drop(['Time    ' ],axis = 1)
#print (whlspd_test_ntime)
print (whlspd_train_ntime.shape)


NN_model = Sequential()
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = whlspd_train_ntime.shape[1], activation='relu'))

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

#Checkpoint callback

#checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
#checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
#callbacks_list = [checkpoint]

NN_model.save('my_model.h5')
#training the model
NN_model.fit(whlspd_train_ntime, VxVeh_train, epochs=500, batch_size=32, validation_split = 0.2)
l1 = []
y_pred = []
y_pred_two_dimns = NN_model.predict(whlspd_test_ntime)
j = 0
for i in range(0,y_pred_two_dimns.shape[0]):
    #print (y_pred_two_dimns[i,j])
    l1.append(y_pred_two_dimns[i,j])
y_pred = l1



prediction = pd.DataFrame({'Time':whlspd_test['Time    ' ],'Actual': VxVeh_test, 'Predicted': y_pred})
print (prediction['Actual'])
print (prediction['Predicted'])

sorted_predictions = prediction.sort_values('Time')

plt.plot(sorted_predictions['Time'],sorted_predictions['Actual'],'b-',label = "True Values"  )
plt.plot(sorted_predictions['Time'],sorted_predictions['Predicted'],'C1',label = "predicted values")
plt.legend(['True Values','predicted values'], loc='upper right')
plt.suptitle('Vehicle speed vs time')
plt.xlabel('time')
plt.ylabel('Vehicle velocity')
