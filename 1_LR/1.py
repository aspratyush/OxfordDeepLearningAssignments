from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential
import numpy as np


''' read test and train data '''
#test#
test_data = np.genfromtxt ('test.csv', delimiter=",")
(rows,cols) = test_data.shape
X_test = np.ones([rows, cols])
X_test[:,1:cols] = test_data[:,1:cols]
Y_test = test_data[:,0]
print "Input read..."

#train#
Y_true = np.genfromtxt ('y_true.csv', delimiter=",")
Y = np.reshape( Y_true, (-1,1) )


''' create the model '''
model = Sequential()
model.add( Dense(input_dim=cols, output_dim=1, init='uniform') )


''' compile '''
sgd = SGD( lr=1e-3, decay=1e-4, momentum=0 )
model.compile( loss='mean_squared_error',optimizer=sgd )


''' fit (train) '''
n_epochs = 100000
model.fit( X_test, Y_test, nb_epoch=n_epochs, verbose=0 )

#save the model
json_string = model.to_json()
open('lin_reg_model.json', 'w').write(json_string)
model.save_weights('lin_reg_weights.h5')
print "Saved model and weights..."


''' evaluate (test) '''
#score = model.evaluate( X_test, Y_true, verbose=0 )
pred = model.predict( X_test )
print "Prediction \t Y_true \t Pred. Error"
print np.hstack([pred,Y,np.abs(pred-Y)])
print "Model weights = ", model.get_weights()


''' LS solution '''
#theta = np.dot( np.linalg.inv( np.dot(np.transpose(X_test), X_test) ), np.dot(np.transpose(X_test),Y_test) )
#print theta
