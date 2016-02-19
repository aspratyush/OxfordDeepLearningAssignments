from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adagrad
import numpy as np
import load_data as ld

#read test and train data
fileName="iris.data.csv"
test_data = ld.IRISLoader(fileName)

'''create the model'''
model = Sequential()
model.add(Dense(input_dim=4, output_dim=4, init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(input_dim=2, output_dim=3, init='uniform'))
model.add(Activation('softmax'))


''' compile '''
adagrad = Adagrad()
model.compile( loss='categorical_crossentropy', optimizer=adagrad )


''' fit(train) '''
n_epochs = 1000
model.fit(nb_epoch=n_epochs, verbose=1)


#save the model
json_string = model.to_json()
open('mlp_model.json', 'w').write(json_string)
model.save_weights('mlp_weights.h5')
print "Saved model and weights..."


