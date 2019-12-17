import numpy as np
import random
import NNlib as nlb

class Neural:
    def __init__(self , data_matrix , batch_size , K_classes, n_hidden = 0 , n_h_neuron = 3 ):
        self.data = data_matrix

        for i in range (len(self.data[0])-1):
            # NORMALIZATION
            # self.data[:,i] = (self.data[:,i] - np.min(self.data[:,i])) / (np.max(self.data[:,i])-np.min(self.data[:,i]))
            
            # STANDARTIZATION 
            self.data[:,i] = (self.data[:,i] - np.mean(self.data[:,i])) / np.std(self.data[:,i])
            

        
        self.n_hidden = n_hidden
        self.n_h_neuron = n_h_neuron
        self.batch_size = batch_size
        self.nbInstances = len(data_matrix)
        self.nbFeatures = len(data_matrix[0])
        self.K_classes = K_classes        
        
        self.trainingSize = int(self.nbInstances * 0.75)
        self.testingSize = self.nbInstances - self.trainingSize
        self.trainingData = np.empty(shape = (self.trainingSize , self.nbFeatures))
        self.testingData = np.empty(shape = (self.testingSize , self.nbFeatures))		
		

        self.W = {}
        self.W[0] = np.empty(shape = (self.nbFeatures - 1, n_h_neuron) , dtype='float64')
        self.W[0] = self.initMatrix(self.W[0])

        self.W[1] = np.empty(shape = (n_h_neuron , K_classes) , dtype='float64')
        self.W[1] = self.initMatrix(self.W[1])


        self.b = {}
        self.b[0] = np.empty(shape = (1 , n_h_neuron))     
        self.b[1] = np.empty(shape = (1 , K_classes))     

		#copy data into training and testing set
        for i in range(self.trainingSize):
            for j in range(self.nbFeatures):    
                self.trainingData[i][j] = self.data[i][j]
				
        for i in range(self.testingSize):
            for j in range(self.nbFeatures):
                self.testingData[i][j] = self.data[i + self.trainingSize][j]
			


        self.X_train = np.empty(shape = (self.batch_size,self.nbFeatures - 1))
        self.Y_train = np.empty(shape = (self.batch_size,self.K_classes))


        # self.X_train = (self.X_train - np.mean(self.X_train)) / np.std(self.X_train)

        self.X_test = self.data[self.trainingSize:, :-1]
        self.Y_test = self.data[self.trainingSize:,  -1]

        # print(self.X_test)
        # self.X_test  = (self.X_test - np.mean(self.X_test)) / np.std(self.X_test)

        one_hot = np.zeros((self.Y_test.shape[0], self.K_classes))

        for i in range(self.Y_test.shape[0]):
            one_hot[i, int(self.Y_test[i])] = 1
        self.Y_test = one_hot





    def initMatrix(self , A):
        self.A = A
        # random.seed(1000)
        for i in range(len(A)):
            for j in range(len(A[0])):  
                self.A[i][j] =  random.uniform(-.0001 , .0001)
                # self.A[i][j] = np.random.normal(0, 1/np.sqrt(self.n_h_neuron)) 

        # print(self.A)
        return self.A     


    def create_one_hot(self , k , indexInBatch , matrixY):
		# print(matrixY,"\n\n")
        for i in range(len(matrixY[0])):
            if i == k:
                matrixY[indexInBatch][i] = 1
            else:
                matrixY[indexInBatch][i] = 0
        # print(matrixY,"\n\n")
        return matrixY


    def load_attributes_labels(self , dataset , X , Y , dataindex ,batch_size):
       
        X = dataset[dataindex : dataindex+batch_size , :-1]

        last_attribute_index = -1
        starting_index = dataindex
        for j in range(batch_size):
            self.create_one_hot(dataset[starting_index + j][last_attribute_index], indexInBatch = j , matrixY = Y)

        return X , Y


    def predict(self, W , X , b):
        H = {}
        H[0] = X @ W[0] + b[0]
        A1 = nlb.NNLib.tanh(H[0])
        H[1] = A1 @ W[1] + b[1]
        y_hat = nlb.NNLib.sigmoid(H[1])

        return y_hat


    def feed_forward(self , X , W, b):
        #using matrix multiplication sign -- @
        H = {}
        H[0] = X @ W[0] + b[0]
        A1 = nlb.NNLib.tanh(H[0])
        H[1] = A1 @ W[1] + b[1]
        y_hat = nlb.NNLib.sigmoid(H[1])
        
        return y_hat , H
        

    def back_prop(self , y_hat , y , H , W ,X):
        dW , db = {} , {}
        loss = 2 * (y_hat - y)
        # when function takes true it means it is derivative of func
        delta = loss * nlb.NNLib.sigmoid(y_hat,True) 

        dW[0] = X.T @ (delta @ W[1].T * nlb.NNLib.tanh(H[0] , True))
        dW[1] = delta.T @ nlb.NNLib.sigmoid(H[0])

        db[0] = delta @ W[1].T * nlb.NNLib.tanh(H[0] , True)
        db[1] = delta

        return dW , db

    
    def prediction_accuracy(self , prev_err , error = False):
        acc = 0
        if error:
            err = np.mean((self.predict(self.W , self.X_test , self.b) - self.Y_test)**2)
            print(prev_err , err)
            
            if prev_err < err:
                # early stopping
                print("early stop")
                return 0
            else:
                prev_err = err
                return prev_err
            return err
        else:
            for i in range(len(self.X_test)):
                acc += nlb.NNLib.accuracy(self.predict(self.W , self.X_test[i] , self.b), self.Y_test[i])

            print(acc / len(self.X_test) * 100)


    
    def train_epoch(self , n_epoch):
        
        epoch = n_epoch
        n_iteration = self.trainingSize/self.batch_size
        prev = 100

        for j in range(n_epoch):
            np.random.shuffle(self.trainingData)
            total_error = 0.

            for i in range(int(n_iteration)):
                self.X_train  , self.Y_train = self.load_attributes_labels(self.trainingData , self.X_train , 
                                        self.Y_train , self.batch_size * i , self.batch_size)

                for z in range(self.batch_size):
                   
                    # ---------------   FEED FORWARD -------------
                    
                    X = self.X_train[z]
                    
                    X = np.reshape(X , (1 ,  X.shape[0]))

                    y_hat , H  = self.feed_forward(X , self.W , self.b)
                    
                    y = self.Y_train[z]

                    
                    # --------------- BACKPROPOGATION

                    error = np.mean((y_hat - y)**2)
                    total_error += error

                    dW , db = self.back_prop(y_hat , y , H , self.W , X )
                                        
                    # ----------UPDATE PARAMETERS -------------
                    n = .01
                    self.W[0] -= n*dW[0]
                    self.W[1] -= n*dW[1].T
                    # print(self.b[0].shape)
                    self.b[0] -= n*db[0]
                    self.b[1] -= n*db[1]


            if j % 10 == 0:
                prev = self.prediction_accuracy(prev , True)
                if prev == 0 and j != 0:
                    break

            # print(total_error / (self.batch_size * n_iteration))



            
            

            


        



        


        
        

