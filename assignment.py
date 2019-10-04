import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from scipy.stats.stats import pearsonr
from math import sqrt


if len(sys.argv) < 3:
	print("Usage is")
	print("python     assignment.py   <input file path>     <output  file path>       <No of split>")
	exit(0)

#Read the input data
df = pd.read_csv(sys.argv[1])

split = int(sys.argv[3])

out = open(sys.argv[2],'w')

final_error = []

size = int(df.shape[0]/split)

matched = 0
relaxed_matched = 0
count = 0
square_sum = 0
sum = 0

#Run the code for each split of input
for i in range(split):
	X = df.loc[i*size: (i+1)*size]
	X = X.astype(float) 
	#Fill the missing values by the average of the column
	X.fillna(X.mean(), inplace=True)
	y = X['y']
	X.drop('y', inplace=True, axis=1)

	X_back = X

	X = X.as_matrix()
	y = y.as_matrix()
	
	#split the data in test and training sample
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state=42)

	
	#normalize the data
	while(1):
		flag = True
		for i in range(X_train.shape[1]):
			if X_train[:,i].std() != 0:
				X_train[:,i] = (X_train[:,i]- X_train[:,i].mean())/X_train[:,i].std()
				X_test[:,i] = (X_test[:,i]- X_test[:,i].mean())/X_test[:,i].std()
			else:
				X_train = np.delete(X_train,i,1)
				X_test = np.delete(X_test,i,1)
				flag = False
				break
		if flag:
			break
		
	av = y_train.mean()
	st = y_train.std()
	y_train = (y_train- y_train.mean())/y_train.std()

	index = []
	i1 = 0
	processed = 0

	#select the columns which is correlated with y
	while(1):
		flag = True
		for i in range(X_train.shape[1]):
			if i > processed :
				i1 = i1 + 1
				corr = pearsonr(X_train[:,i], y_train)
				PEr= .674 * (1- corr[0]*corr[0])/ (len(X_train[:,i])**(1/2.0))
				if abs(corr[0]) < PEr:
					X_train = np.delete(X_train,i,1)
					X_test = np.delete(X_test,i,1)
					index.append(X_back.columns[i1-1])
					processed = i - 1 
					flag = False
					break
		if flag:
			break
	#drop the columns which is correlated with other input column
	while(1):
		flag = True
		for i in range(X_train.shape[1]):
			for j in range(i+1,X_train.shape[1]-1):
				corr = pearsonr(X_train[:,i], X_train[:,j])
				PEr= .674 * (1- corr[0]*corr[0])/ (len(X_train[:,i])**(1/2.0))
				if abs(corr[0]) > 6*PEr:
					X_train = np.delete(X_train,j,1)
					X_test = np.delete(X_test,j,1)
					flag = False
					break
			break
		if flag:
			break
		
		
	#build the model to predict the y	
	learning_rate = 0.0001

	model = Sequential([
		Dense(64, activation=tf.nn.relu, input_shape=[X_train.shape[1]]),
		Dense(64, activation=tf.nn.relu),
		Dense(1)
	])
  
	optimizer = tf.train.RMSPropOptimizer(learning_rate)

	model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
				
	model.fit(
	X_train, y_train,
	epochs=int(X_train.shape[1]/2), validation_split = 0.2, verbose=0)
  
	predict = model.predict(X_train)
	
	#build the model to predict the error in prediction
	error = []	
	for i in range(len(predict)):
		error.append(y_train[i] - predict[i][0])
	
	error = np.array(error)

	model_e = Sequential([
		Dense(64, activation=tf.nn.relu, input_shape=[X_train.shape[1]]),
		Dense(64, activation=tf.nn.relu),
		Dense(1)
	])
  
	model_e.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
				
	model_e.fit(
	X_train, error,
	epochs=int(X_train.shape[1]/2), validation_split = 0.2, verbose=0)
  

	#predict the test data using the trained model
	predict = model.predict(X_test) 
 
	err_p = model_e.predict(X_test)
	 
	predict = predict + err_p
 
	predict = predict*st + av
 
	for i in range(len(predict)):
		error = y_test[i] - predict[i][0]
		if abs(error) <= 3:
			matched = matched + 1
		if abs(error/y_test[i]) <= 0.1:
			relaxed_matched = relaxed_matched + 1
		square_sum = square_sum + error*error
		sum = sum + error
		count = count + 1
		
out.write("RMSE="+str(sqrt(square_sum/count))+'\n')
out.write("matched count="+ str(matched) +'\t Total count=' + str(count) +'\n')

out.write("ME="+str(sqrt(abs(sum)/count))+'\n')
out.write("relaxed matched count="+ str(relaxed_matched) +'\t Total count=' + str(count) +'\n')

out.close()


print("RMSE=",str(sqrt(square_sum/count)),'\n')
print("matched count=", str(matched),'\t', "Total count=", str(count),'\n')

print("ME=",str(sqrt(abs(sum)/count)),'\n')
print("relaxed matched count=", str(relaxed_matched),'\t', "Total count=", str(count),'\n')
