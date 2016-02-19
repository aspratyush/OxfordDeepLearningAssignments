import csv
import numpy as np

'''classes in data'''
classNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
classLabels = [0,1,2]


'''IRISLoader API'''
def IRISLoader( fileName ):
	
	#get data using np
	test_data = np.genfromtxt (fileName, delimiter=",")
	#print test_data

	#handle last column
	f = open(fileName, 'rb')
	#iterable over file
	reader = csv.reader(f)
	#read each line
	nLineNo = 0
	for row in reader:
		#print row[-1], classNames
		if ( len(row) > 0 ):
			nLabel = classNames.index(row[-1])
			test_data[nLineNo,-1] = nLabel
			nLineNo += 1
	
	#print "---------Modified data---------"
	np.random.seed(1)
	np.random.shuffle(test_data)
	return test_data

