import Neural_network as NN
import Data_manipulation as Dp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NNlib as nlb

path = "disease.csv"



# ANALYSING THE DATA 

data = pd.read_csv(path , sep=";")
# nlb.NNLib.correlation(data)

categorical_columns = ['chest_pain_type' , 'fasting_blood_sugar' , 'rest_ecg' , 'exercise_induced_angina' , 'st_slope' , 
    'num_major_vessels' ,'thalassemia']

# we will make our categorical columns in one hot structure so that it will be much better to train for our network
data = pd.get_dummies(data , columns = categorical_columns , prefix = categorical_columns) 



matrix = data.to_numpy()
np.random.shuffle(matrix)
nn = NN.Neural(data_matrix=matrix ,batch_size = 4 , K_classes = 2 , n_hidden=1 , n_h_neuron=5)


history = nn.train_epoch(n_epoch = 10)


y_pred = nn.predict(nn.W , nn.X_test , nn.b)

nlb.NNLib.plot(history)


print(f"Accuracy    : {nn.accuracy(y_pred , nn.Y_test)}")
print(f"Recall      : {nn.recall(y_pred , nn.Y_test)}")
print(f"Precision   : {nn.precision(y_pred , nn.Y_test)}")
print(f"Specificity :{nn.specificity(y_pred,nn.Y_test)}")
print(f"F1 score    : {nn.f1_score(y_pred , nn.Y_test)}")
