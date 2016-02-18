from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np

'''
#read test and train data
#test data#
test_data = np.genfromtxt ('test.csv', delimiter=",")
(rows,cols) = test_data.shape
X_test = np.ones([rows, cols])
X_test[:,1:cols] = test_data[:,1:cols]
Y_test = test_data[:,0]
print "Input read..."

#training data#
Y_true = np.genfromtxt ('y_true.csv', delimiter=",")
Y = np.reshape( Y_true, (-1,1) )
'''

'''create the model'''
model = Sequential()
model.add( Dense(input_dim=4, output_dim=4, init='uniform') )
