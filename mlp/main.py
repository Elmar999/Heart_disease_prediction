import Neural_network as NN
import Data_manipulation as Dp
import pandas as pd
import numpy as np

path = "disease.csv"


data = pd.read_csv(path , sep=";")



# data = data.drop(columns = [ 'st_depression' ])
mat_corr = data.corr()
# print(mat_corr)


# print(data.head())
matrix = data.to_numpy()
# print(matrix)
# print()

np.random.shuffle(matrix)
# print(matrix)
nn = NN.Neural(data_matrix=matrix ,batch_size = 4 , K_classes = 2 , n_hidden=1 , n_h_neuron=128)


nn.train_epoch(n_epoch = 300)
err = 0
nn.prediction_accuracy(err)

# print(nn.Y_train)

