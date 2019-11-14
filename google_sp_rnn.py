import os

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt


os.environ["KERAS_BACKEND"] = "tensorflow"

training_data=pd.read_csv('Google_Stock_price_Train.csv')
training_data=training_data.iloc[:,1:2].values


from sklearn.preprocessing import MinMaxScaler
# we will go for  Normalisation as it will keep in [0,1] range.. as we use sigmoid fx.It ouputs [0,1] range

sc=MinMaxScaler() # range(0,1)

training_data=sc.fit_transform(training_data) # transforming in between 0-1 range

X_train=training_data[0:1257]
y_train=training_data[1:1258]

# you almost always need to reshape the data
# as you have to add time var also
# keras tensor __doc__ says 3D Tensor as this to (num_of_input,time,input_dim)
# num_of_input=1257 time=1day gap & input_dim=1

X_train=np.reshape(X_train,(1257,1,1))

from keras.models import Sequential
from keras.layers import Dense,LSTM

# initialising the rnn

# t+1 ouput for t input

# regression for continuous var and classification for category
# ann and cnn are classification net and rnn is a regression net
# in Boltzman Machine we will use the Graph Layer


regressor=Sequential()
# 4 memory cell...activation----tanh/sigmoid...input shape(time can be any dur,feature input dim)
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))

regressor.add(Dense(units=1))
# rmsprop > adam...memory ....and rmsprop works great than rmsprop in rnn
regressor.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])

regressor.fit(X_train,y_train,batch_size=32,epochs=200)



test_set=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test_set.iloc[:,1:2].values

# we are going to predict one day t+1 with data of one day t
# we have to wait one day
# as of this dataset what it will do is
'''
when you input january 1st,2017 stock price then it will predict
for January 2nd,2017.And so on.Then for Jan 2 will be for 3.

For 20 days it can predict 19 days.How?

--- 1st day predict 2nd. Now 2nd day you know 2nd so predict
3rd. In 3rd day you know 3rd so predict 4th.

Well here the structure stands as t=1 so we are going to predict only

1 (one) time step ahead of us. If ypu need more then just input

timestep value in the tensor. 

It is a one2one LSTM/RNN Structure. But for M2M like texts it might
be used more timestep



'''
inputs=real_stock_price
inputs=sc.transform(inputs)
inputs=np.reshape(inputs,(20,1,1))

predicted_sp=regressor.predict(inputs)
predicted_sp=sc.inverse_transform(predicted_sp)

plt.plot(real_stock_price,color='red',label='Real Google Stock Price(Linearaxon AI SYS)')
plt.plot(predicted_sp,color='blue',label='LinearAxon Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
plt.savefig('linearaxon_stock_p.png')

# error:


# evaluate regression model....rmse=root_mean_square_error of test set...
import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price,predicted_sp))

print('The error is: %.2f%%' %( (rmse/np.average(real_stock_price)))*100 )
print('The Accuracy is %.2f%%' %(100- (rmse/np.average(real_stock_price)))*100 )