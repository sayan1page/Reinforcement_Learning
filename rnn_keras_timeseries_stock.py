'''
Created on Aug 1, 2017

'''
#import matplotlib.pyplot as plt
import numpy as np
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import sys


def read_data(path_to_dataset, path_to_target,
                           sequence_length=50,
                           ratio=1.0):

	max_values = ratio * 2049280

	with open(path_to_dataset) as f:
		data = csv.reader(f, delimiter=",")
		power = []
		nb_of_values = 0
		
		for line in data:
			try:
				power.append([float(line[1]),float(line[4]),float(line[7])])
				nb_of_values += 1
			except ValueError:
				pass
			if nb_of_values >= max_values:
				break
				
	with open(path_to_target) as f:
		data = csv.reader(f, delimiter=",")
		target = []
		nb_of_values = 0
		for line in data:
			try:
				target.append(float(line[0].strip()))
				nb_of_values += 1
			except ValueError:
				pass
			if nb_of_values >= max_values:
				break
	return power, target


def create_matrix(y_train):
	y = [[0 for i in range(3)] for j in range(len(y_train))]
	for i in range(len(y_train)):
		if y_train[i] == -100:
			y[i][0] = 1
		else:
			if y_train[i] == 100:
				y[i][1] = 1
			else:
				if y_train[i] == 0:
					y[i][2] = 1
	return y
		

def process_data(power, target, sequence_length):
	result = []
	
	for index in range(len(power) - sequence_length-1):
		result.append(power[index: index + sequence_length])
	result = np.array(result) 
	
	#print(result.shape)
	
	row = int(round(0.9 * result.shape[0]))
    
	X_train = result[:row, :]
	
	#X_train = train[:, :-1]
	
	y_train = np.array(create_matrix(target))
	#print(y_train.shape)
	X_test = result[row:, :]
	y_test = y_train[row:]
	#print(y_test.shape)
	y_train = y_train[:row]
	#print(y_train.shape)
	
	#print(X_train.shape)

	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 3))
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 3))

	return [X_train, y_train, X_test, y_test]


def build_model():
	model = Sequential()
	layers = [3, 100, 50, 3]

	model.add(LSTM(
		layers[1],
		input_shape=(None, layers[0]),
		return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(
		layers[2],
		return_sequences=False))
	model.add(Dropout(0.2))

	model.add(Dense(
		layers[3]))
	model.add(Activation('softmax'))

	model.compile(loss="categorical_crossentropy", optimizer="adam")
    
	return model


def run_network(data=None, target=None):
	epochs = 2
	ratio = 0.5
	sequence_length = 50

	X_train, y_train, X_test, y_test = process_data(
		data, target, sequence_length)

	model = build_model()

	try:
		model.fit(
			X_train, y_train,
			batch_size=512, nb_epoch=epochs, validation_split=0.05, verbose=0)
		predicted = model.predict(X_test)
	except KeyboardInterrupt:
		exit(0)

	return y_test, predicted
	
def convert(x):
	if x[0] == 1:
		return -100
	if x[1] == 1:
		return 100
	if x[2] == 1:
		return 0


		
if __name__ == '__main__':
	path_to_dataset = 'TrainingFeatureBid.data'
	path_to_target = 'TrainingTargetBid.data'
	data, target = read_data(path_to_dataset, path_to_target)
	k_inc = 1
	k_dec = 1
	k_same = 1
	
	for i in range(0,len(data)-1000,89):
		d = data[i:i+1001]
		t = target[i:i+1001]
		y_test, predicted = run_network(d,t)
		
		prob_increasing = predicted[:,1]
		increasing_mean = prob_increasing.mean()
		increasing_std = prob_increasing.std()
		prob_decreasing = predicted[:,0]
		decreasing_mean = prob_decreasing.mean()
		decreasing_std = prob_decreasing.std()
		prob_same = predicted[:,2]
		same_mean = prob_same.mean()
		same_std = prob_same.std()
		wrong_count_up = 0
		total_count_up = 0
		wrong_count_pos_up = 0
		total_count_pos_up = 0
		wrong_count_neg_up = 0
		total_count_neg_up = 0
		wrong_count_down = 0
		total_count_down = 0
		wrong_count_pos_down = 0
		total_count_pos_down = 0
		wrong_count_neg_down = 0
		total_count_neg_down = 0
		for j in range(len(predicted)-1):
			inc = (prob_increasing[j] - increasing_mean + k_inc*increasing_std)
			dec = (prob_decreasing[j] - decreasing_mean + k_dec*decreasing_std)
			same = (prob_same[j] - same_mean +  k_same*same_std)
			acc_status = convert(y_test[j])
			if same > 0:
				pr_status = 0
			else:
				if inc > dec:
					pr_status = 100
				else:
					pr_status = -100
			
			if acc_status == 0:
				if pr_status != acc_status:
					wrong_count_up = wrong_count_up + 1
				total_count_up = total_count_up + 1
			else:
				if acc_status == 100:
					if pr_status != acc_status:
						wrong_count_pos_up = wrong_count_pos_up + 1
					total_count_pos_up = total_count_pos_up + 1
				else:
					if pr_status != acc_status:
						wrong_count_neg_up = wrong_count_neg_up + 1
					total_count_neg_up = total_count_neg_up + 1
			
			if pr_status == 0:
				if pr_status != acc_status:
					wrong_count_down = wrong_count_down + 1
				total_count_down = total_count_down + 1
			else:
				if pr_status == 100:
					if pr_status != acc_status:
						wrong_count_pos_down = wrong_count_pos_down + 1
					total_count_pos_down = total_count_pos_down + 1
				else:
					if pr_status != acc_status:
						wrong_count_neg_down = wrong_count_neg_down + 1
					total_count_neg_down = total_count_neg_down + 1
				
				
			print(acc_status,',', pr_status)
		
		if total_count_up !=0:
			if wrong_count_up/total_count_up > .5:
				k_same = 1.2 * k_same
		if total_count_down !=0:
			if wrong_count_down/total_count_down > .5:
				k_same = 0.9 * k_same
						
		if total_count_pos_up !=0:
			if wrong_count_pos_up/total_count_pos_up > .5:
				k_inc = 1.2 * k_inc
		if total_count_pos_down !=0:
			if wrong_count_pos_down/total_count_pos_down > .5:
				k_inc = 0.9 * k_inc
			
		if total_count_neg_up !=0:
			if wrong_count_neg_up/total_count_neg_up > .5:
				k_dec = 1.2 * k_dec
		if total_count_neg_down !=0:
			if wrong_count_neg_down/total_count_neg_down > .5:
				k_dec = 0.9 * k_dec
				
				