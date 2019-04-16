import numpy as np

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=False):
    # Calculate number of batches possible
    losses = []
    num_batches = int(len(x_train)/200)
    #IMPLEMENT HERE
    for epochs in range(epoch):
        print(epochs)
        loss = 0
        if shuffle:
            p = np.random.permution(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]
        for batches in range(num_batches):
            X = x_train[((batches%num_batches)*200):((batches%num_batches)*200)+199]
            y = y_train[((batches%num_batches)*200):((batches%num_batches)*200)+199]
            loss += four_nn(w1, w2, w3, w4, b1, b2, b3, b4, X, y, num_classes)
        losses.append(loss)
    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    learning_rate = .1

    z1, acache1 = affine_forward(x_test, w1, b1)
    a1, rcache1 = relu_forward(z1)
    z2, acache2 = affine_forward(a1, w2, b2)
    a2, rcache2 = relu_forward(z2)
    z3, acache3 = affine_forward(a2, w3, b3)
    a3, rcache3 = relu_forward(z3)

    #Reach final layer of NN
    F, acache4 = affine_forward(a3, w4, b4)

    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    class_counts= [0.0] * num_classes
    classification = []
    
    n = len(y_test)
    
    # Classify the images
    for i in range(F.shape[0]):
        classification.append(np.argmax(F[i,:]))
    # Test the classifications
    for j in range(n):
        if classification[j] == y_test[j]:
            avg_class_rate += 1
            class_rate_per_class[int(y_test[j])] += 1
        class_counts[int(y_test[j])] += 1
    
    avg_class_rate = avg_class_rate/n
    
    for i in range(num_classes):
        class_rate_per_class[i] = (class_rate_per_class[i])/class_counts[int(y_test[j])]
   
    
    return avg_class_rate, class_rate_per_class, classification

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    learning_rate = .1

    z1, acache1 = affine_forward(x_test, w1, b1)
    a1, rcache1 = relu_forward(z1)
    z2, acache2 = affine_forward(a1, w2, b2)
    a2, rcache2 = relu_forward(z2)
    z3, acache3 = affine_forward(a2, w3, b3)
    a3, rcache3 = relu_forward(z3)

    #Reach final layer of NN
    F, acache4 = affine_forward(a3, w4, b4)

    loss, dF = cross_entropy(F,y_test)
    #Backpropogate
    dA3, dW4, db4 = affine_backward(dF, acache4)
    dZ3 = relu_backward(dA3, rcache3)
    dA2, dW3, db3 = affine_backward(dZ3, acache3)
    dZ2 = relu_backward(dA2, rcache2)
    dA1, dW2, db2 = affine_backward(dZ2, acache2)
    dZ1 = relu_backward(dA1, rcache1)
    dX, dW1, db1 = affine_backward(dZ1, acache1)

    # Use gradient descent to update weights
    w1 -= (learning_rate)*dW1
    w2 -= (learning_rate)*dW2
    w3 -= (learning_rate)*dW3
    w4 -= (learning_rate)*dW4

    return loss

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    # Save cache
    cache = (A,W,b)
    Z = np.matmul(A,W)
    # Add Bias
    Z = np.add(Z, b)
    return Z, cache

def affine_backward(dZ, cache):
    # Rowise addition with axis = 0
    dB = np.sum(dZ, 0)
    # Transpose to make shapes work
    dW = np.matmul(cache[0].transpose(), dZ)
    dA = np.matmul(dZ, cache[1].transpose())
    return dA, dW, dB

def relu_forward(Z):
    A = np.array(Z)
    A[Z<0] = 0
    return A, Z

def relu_backward(dA, cache):
    dZ = np.zeros((dA.shape[0], dA.shape[1]))
    # Where dA is >0 make that the dZ value
    dZ[tuple([cache > 0])] = dA[tuple([cache > 0])]
    return dZ

def cross_entropy(F, y):
    n = len(y)
    num_classes = F.shape[1]
    F_e = np.exp(F)
    
    actual_score = 0
    predicted_score = 0
    for i in range(n):
        actual_score += F[int(i)][int(y[i])]
        predicted_score += np.log(np.sum(F_e[int(i)]))

    loss = (-1/n)*(actual_score - predicted_score)

    dF = np.zeros(F.shape)
    for i in range(n):
        normalization = np.sum(F_e[int(i)])
        for j in range(num_classes):
            dF[i][j] = (-1/n)*((1 if j == y[i] else 0) - (F_e[int(i)][int(j)]/normalization)) 

    return loss, dF
