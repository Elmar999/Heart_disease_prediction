import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class NNLib:

    # def initMatrix(self , A):
    #     self.A = A
    #     random.seed(100)
    #     for i in range(len(A)):
    #         for j in range(len(A[0])):
    #             self.A[i][j] =  random.uniform(-.0001 , .0001)

    #     print(self.A)
    #     return self.A     


    # activation function - ReLU
    # Z - matrix of activated weighted input
    # return activated matrix
    def relu(Z):
        activation = np.empty(shape = Z.shape)
        # print(activation.shape)
        for i in range(len(Z)):
            for j in range(len(Z[0])):
                activation[i][j] = Z[i][j] if Z[i][j] > 0 else 0
        return activation


    def softmax(Z):
        
        # print(Z)

        # print("\n\n")
        # softA = np.empty(shape = Z.shape)
        # for i in range(len(softA[0])):
        #     for j in range(len(softA)):
        #         s = 0
        #         for c in range(len(softA)):
        #             s += np.exp(Z[c][i])
        #         softA[j][i] = np.exp(Z[j][i])/s

        # return softA

        e_x = np.exp(Z - np.max(Z))

        return e_x / e_x.sum()



    def tanh(x , derivative = False):
        if not derivative:
            return (2/(1+np.exp(-x))) - 1

        return 1 - ((2/(1+np.exp(-x))) - 1)**2


    def sigmoid(x, derivative=False):
        if not derivative:
            # x = np.clip( x, -500, 500 )
            return 1 / (np.exp(-x) + 1)
        # return x * (1 - x)
        return np.exp(-x)*(1 / (np.exp(-x) + 1)**2)



    def crossEntropy(yHat , y):
        K = len(y)
        cost = 0.0
        batchSize = len(yHat[0])
        for k in range(K):
            for c in range(batchSize):
                # if yHat[k][c] == 0:
                #     print("\nnnlib\n ", yHat , y)
                #     cost += 0
                #     exit(0)
                #     continue
                cost += y[k][c]*np.log(yHat[k][c] + 1e-9)
        
        return -(1.0/batchSize)*cost
        # return -np.log(yHat[np.where(y)])


    def binary_cross_entropy(self , y_pred , y_true):
        res = 0

        pass    


    def softmax_deriv(Z):
        return Z*(1-Z)


    def hadamard(A , B):
        C = np.empty(shape=(len(A) , len(A[0])))
        for i in range(len(A)):
            for j in range(len(A[0])):
                C[i][j] = A[i][j] * B[i][j]
        return C


    def relu(H):
        activations = np.empty(shape = (len(H) , len(H[0])))
        for i in range(len(H)):
            for j in range(len(H[0])):
                activations[i][j] = H[i][j] if H[i][j] > 0 else 0
        return activations

    def relu_deriv(A):
        C = np.empty(shape=(len(A) , len(A[0])))
        for i in range(len(A)):
            for j in range(len(A[0])):
                C[i][j] = 1 if A[i][j] > 0 else 0

        return C


    def accuracy(y_pred , y_true):
        # print(y_pred , y_true)
        y_pred = np.argmax(y_pred)
        y_true = np.argmax(y_true)

        return np.sum(y_pred == y_true)

    
    def plot(train_loss , test_loss ):
        # print(train_loss)
        plt.plot(train_loss)
        plt.plot(test_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()



    def confusion_matrix(y_hat , y_true):
        
        #Tp -- Fp -- Fn -- Tn
        matrix = [0, 0, 0, 0]
        # print(y_pred)
        for i in range(len(y_hat)):
            if   np.argmax(y_hat[i]) == 0 and np.argmax(y_true[i]) == 0:
                matrix[0] += 1 # true positive
            elif np.argmax(y_hat[i]) == 0 and np.argmax(y_true[i]) == 1:
                matrix[1] += 1 # false positive
            elif np.argmax(y_hat[i]) == 1 and np.argmax(y_true[i]) == 1:
                matrix[3] += 1 # true negative
            else:
                matrix[2] += 1 # false negative 
        
        return np.array(matrix)


    def correlation(data):
        mat_corr = data.corr()
        f, ax = plt.subplots(figsize =(9, 8)) 
        sns.heatmap(mat_corr, ax = ax, annot = True , cmap ="YlGnBu", linewidths = 0.1) 
        plt.show()