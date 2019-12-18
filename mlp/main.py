import Neural_network as NN
import Data_manipulation as Dp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NNlib as nlb

path = "disease.csv"


data = pd.read_csv(path , sep=";")


categorical_columns = ['chest_pain_type' , 'fasting_blood_sugar' , 'rest_ecg' , 'exercise_induced_angina' , 'st_slope' , 
    'num_major_vessels' ,'thalassemia']

data = pd.get_dummies(data , columns = categorical_columns , prefix = categorical_columns)

mat_corr = data.corr()
matrix = data.to_numpy()

np.random.shuffle(matrix)

nn = NN.Neural(data_matrix=matrix ,batch_size = 4 , K_classes = 2 , n_hidden=1 , n_h_neuron=5)



error_train , error_test = [] , []
error_train  , error_test = nn.train_epoch(n_epoch = 50)

nn.prediction_accuracy()

nlb.NNLib.plot(error_train , error_test)




# print(nn.Y_train)

