import Neural_network as NN
import Data_manipulation as Dp
import pandas as pd
import numpy as np

path = "disease.csv"


data = pd.read_csv(path , sep=";")


categorical_columns = []

for col in list(data):
    # print(len(data[col].unique()))
    if len(data[col].unique()) < 4:
        pd.get_dummies(data , columns = col , prefix = col)


mat_corr = data.corr()
# print(mat_corr)


# print(data.head())
matrix = data.to_numpy()
# print(matrix)
# print()

np.random.shuffle(matrix)
# print(matrix)
nn = NN.Neural(data_matrix=matrix ,batch_size = 4 , K_classes = 2 , n_hidden=1 , n_h_neuron=5)


nn.train_epoch(n_epoch = 500)
err = 0
nn.prediction_accuracy(err)

# print(nn.Y_train)

