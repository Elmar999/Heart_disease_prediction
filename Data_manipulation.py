import numpy as np
import pandas as pd

class Data_manip:
	def __init__(self , path ):
		self.data_matrix = self.convert_data_into_2Dmatrix(path)
		# print(self.data_matrix)
          
	def convert_data_into_2Dmatrix(self, path):
		
		file = open(path , 'r')
		data = file.readlines()
		nbRows = len(data)
		nbCols = len(data[0].split(";"))

		print(nbCols , nbRows)

		content = [x.split(';') for x in data]


		arr = np.empty(shape = (nbRows , nbCols)) 
		
		for i in range(len(arr)):
			for j in range(len(arr[0])):
				arr[i][j] = content[i][j]
		
		# arr[:, :4] = (arr[:, :4] - arr[:, :4].mean(axis=0)) / arr[:, :4].std(axis=0)

		#return np.random.shuffle(arr) 
		# print(type(arr))
		np.random.shuffle(arr)
		return arr
    
    
	def get_matrix(self):
		return self.data_matrix
    
	






