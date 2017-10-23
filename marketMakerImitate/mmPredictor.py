
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Normalize array by subtracting mean and dividing by std dev
def normalize(array): 
    array=(array - np.mean(array,axis=0,keepdims=True ))/np.std(array,axis=0,keepdims=True)
    return array

if __name__ == '__main__':

    input_file = "inputs.csv"
    output_file = "outputs.csv"

    #Load input data
    df = pd.read_csv(input_file, header = None)
    # remove the non-numeric columns
    df = df._get_numeric_data()
    X = df.as_matrix()
    print(X[0:10,:])
    X = normalize(X)
    print(X[0:10,:])

    #Load output data
    df = pd.read_csv(output_file, header=None)
    df = df._get_numeric_data()
    y = df.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("training samples: " + str(X_train.shape))
    print("test samples: " + str(X_test.shape))
    #print(X_train[0:10,:])

    mlp = MLPClassifier(verbose='True', hidden_layer_sizes=(8), solver='sgd', batch_size='auto', learning_rate_init=0.01, activation='logistic', max_iter=1000)
    mlp.fit(X_train, y_train)

    score = mlp.score(X_test, y_test)
    print(score)

    info = 'Score: {0}\nIterations: {1} Learning Rate: {2} Activation: {3} Layers: {4}' .format(score, mlp.n_iter_, mlp.learning_rate_init, mlp.activation, mlp.hidden_layer_sizes)

    fig = plt.figure()
    plt.plot(mlp.loss_curve_)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.figtext(0.01,0.001,info)
    fig.suptitle('With feature normalization')
    
    plt.show()
     
    print("finished " + str(mlp.n_iter_) + " iterations")
    print("Output activaiton: " +  mlp.out_activation_)
    print("Number outputs: " + str(mlp.n_outputs_))

    #print(mlp.coefs_)
    # test1 = [0,0,0,0.8,0,0,0,0,0]
    #print(str(mlp.predict_proba(test1)) + "," + str(mlp.predict(test1))) #expect 0,0,1
    