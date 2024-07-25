import pandas as pd
import cupy as cp
from matplotlib import pyplot as plt
import os

def init_params(input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
    W1 = cp.random.randn(hidden1_size, input_size) * cp.sqrt(1. / hidden1_size)
    b1 = cp.zeros((hidden1_size, 1))
    W2 = cp.random.randn(hidden2_size, hidden1_size) * cp.sqrt(1. / hidden2_size)
    b2 = cp.zeros((hidden2_size, 1))
    W3 = cp.random.randn(hidden3_size, hidden2_size) * cp.sqrt(1. / hidden3_size)
    b3 = cp.zeros((hidden3_size, 1))
    W4 = cp.random.randn(output_size, hidden3_size) * cp.sqrt(1. / output_size)
    b4 = cp.zeros((output_size, 1))
    return W1, b1, W2, b2, W3, b3, W4, b4

def ReLU(Z):
    return cp.maximum(0, Z)

def Deriv_ReLU(Z):
    return Z > 0

def Tanh(Z):
    return cp.tanh(Z)

def Deriv_Tanh(Z):
    return 1 - cp.tanh(Z)**2

def Softmax(Z):
    expZ = cp.exp(Z - cp.max(Z, axis=0, keepdims=True))
    return expZ / expZ.sum(axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = Tanh(Z2)
    
    Z3 = W3.dot(A2) + b3
    A3 = ReLU(Z3)
    
    Z4 = W4.dot(A3) + b4
    A4 = Softmax(Z4)
    
    return Z1, A1, Z2, A2, Z3, A3, Z4, A4

def One_hot(Y, num_classes):
    Y_vec = cp.zeros((Y.size, num_classes))
    Y_vec[cp.arange(Y.size), Y] = 1
    return Y_vec.T

def back_prop(Z1, A1, W2, Z2, A2, W3, Z3, A3, W4, Z4, A4, X, Y):
    m = Y.size
    num_classes = A4.shape[0]
    Y_hot = One_hot(Y, num_classes)
    
    dZ4 = A4 - Y_hot
    dW4 = 1 / m * dZ4.dot(A3.T)
    db4 = 1 / m * cp.sum(dZ4, axis=1, keepdims=True)
    
    dZ3 = W4.T.dot(dZ4) * Deriv_ReLU(Z3)
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * cp.sum(dZ3, axis=1, keepdims=True)
    
    dZ2 = W3.T.dot(dZ3) * Deriv_Tanh(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * cp.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) * Deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * cp.sum(dZ1, axis=1, keepdims=True)
    
    return db1, dW1, db2, dW2, db3, dW3, db4, dW4

def update_params(W1, b1, W2, b2, W3, b3, W4, b4, db1, dW1, db2, dW2, db3, dW3, db4, dW4, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    
    W2 -= alpha * dW2
    b2 -= alpha * db2
    
    W3 -= alpha * dW3
    b3 -= alpha * db3
    
    W4 -= alpha * dW4
    b4 -= alpha * db4
    
    return W1, b1, W2, b2, W3, b3, W4, b4

def save_params(W1, b1, W2, b2, W3, b3, W4, b4, folder_path='params'):
    cp.savetxt(os.path.join(folder_path, 'W1.txt'), W1.get())
    cp.savetxt(os.path.join(folder_path, 'b1.txt'), b1.get())
    cp.savetxt(os.path.join(folder_path, 'W2.txt'), W2.get())
    cp.savetxt(os.path.join(folder_path, 'b2.txt'), b2.get())
    cp.savetxt(os.path.join(folder_path, 'W3.txt'), W3.get())
    cp.savetxt(os.path.join(folder_path, 'b3.txt'), b3.get())
    cp.savetxt(os.path.join(folder_path, 'W4.txt'), W4.get())
    cp.savetxt(os.path.join(folder_path, 'b4.txt'), b4.get())

def get_predictions(A):
    return cp.argmax(A, axis=0)

def get_accuracy(predictions, Y):
    return cp.mean(predictions == Y)

def Train(X, Y, input_size, hidden1_size, hidden2_size, hidden3_size, output_size, epochs, alpha):
    W1, b1, W2, b2, W3, b3, W4, b4 = init_params(input_size, hidden1_size, hidden2_size, hidden3_size, output_size)
    for i in range(1, epochs + 1):
        Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
        db1, dW1, db2, dW2, db3, dW3, db4, dW4 = back_prop(Z1, A1, W2, Z2, A2, W3, Z3, A3, W4, Z4, A4, X, Y)
        W1, b1, W2, b2, W3, b3, W4, b4 = update_params(W1, b1, W2, b2, W3, b3, W4, b4, db1, dW1, db2, dW2, db3, dW3, db4, dW4, alpha)
        
        if i % 20 == 0:
            predictions = get_predictions(A4)
            accuracy = get_accuracy(predictions, Y) * 100
            print(f"Iteration: {i}, Accuracy: {accuracy:.4f}")
    
    return W1, b1, W2, b2, W3, b3, W4, b4

data = cp.array(pd.read_csv('data.csv').values)
m, n = data.shape
data = data[cp.random.permutation(m)]

data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.0

input_size = 784
hidden1_size = 128
hidden2_size = 64
hidden3_size = 32
output_size = 10 

epochs = 1000
alpha = 0.3

W1, b1, W2, b2, W3, b3, W4, b4 = Train(X_train, Y_train, input_size, hidden1_size, hidden2_size, hidden3_size, output_size, epochs, alpha)
save_params(W1, b1, W2, b2, W3, b3, W4, b4)