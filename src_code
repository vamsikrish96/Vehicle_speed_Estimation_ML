import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Vehicle data taken from Carsim

vehicle_data = pd.read_csv("U:\\Vehicle_spd_Estimation\\vehicle_data_dataset.csv")

print(vehicle_data.columns.tolist())

#Data preparation for training

fin_vehicle_data = vehicle_data.drop(['VxVeh_kph' ],axis = 1)
for i in range(0,11):
    fin_vehicle_data.drop(fin_vehicle_data.index[0],inplace = True)
#print (fin_vehicle_data)
plt.scatter(fin_vehicle_data['VxWhl1'],fin_vehicle_data['VxVeh_ms'])
plt.show()
Wheel_speed_feature = fin_vehicle_data.drop(['VxVeh_ms'],axis = 1)
vehicle_speed_label = fin_vehicle_data['VxVeh_ms']
whlspd_train, whlspd_test, VxVeh_train, VxVeh_test = train_test_split(Wheel_speed_feature, vehicle_speed_label, test_size = 0.2, random_state = 0)

#Using linear regression for training the data

regr = LinearRegression()
whlspd_train_ntime = whlspd_train.drop(['Time    ' ],axis = 1)
whlspd_test_ntime = whlspd_test.drop(['Time    ' ],axis = 1)
regr.fit(whlspd_train_ntime, VxVeh_train)
y_pred = regr.predict(whlspd_test_ntime)
prediction = pd.DataFrame({'Time':whlspd_test['Time    ' ],'Actual': VxVeh_test, 'Predicted': y_pred})
print(whlspd_train_ntime.head()) #features train data
print(whlspd_test_ntime.head())  #labels test data

print (whlspd_test['Time    '].head()) 
print (prediction['Actual'].head()) #Prediction values
sorted_predictions = prediction.sort_values('Time')

#Plotting the graph using matplotlib for comparison of original signal and predicted signal
 
plt.plot(sorted_predictions['Time'],sorted_predictions['Actual'],'b-',label = "True Values"  )

plt.plot(sorted_predictions['Time'],sorted_predictions['Predicted'],'C1',label = "predicted values")
plt.legend(['True Values','predicted values'], loc='upper right')
plt.suptitle('Vehicle speed vs time')
plt.xlabel('time')
plt.ylabel('Vehicle velocity')
mean_squared_error(prediction['Actual'], prediction['Predicted'])
print (mean_squared_error(prediction['Actual'], prediction['Predicted']))

#The mean square is predicted 
