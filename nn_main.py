from neural_network import minibatch_gd, test_nn
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def init_weights(d, dp):
    return 0.01 * np.random.uniform(0.0, 1.0, (d, dp)), np.zeros(dp)

start_time = time.time()
if __name__ == '__main__':
    x_train = np.load("data/x_train.npy")
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    y_train = np.load("data/y_train.npy")

    x_test = np.load("data/x_test.npy")
    x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
    y_test = np.load("data/y_test.npy")

    load_weights = False #set to True if you want to use saved weights

    if load_weights:
        w1 = np.load('w1')
        w2 = np.load('w2')
        w3 = np.load('w3')
        w4 = np.load('w4')

        b1 = np.load('b1')
        b2 = np.load('b2')
        b3 = np.load('b3')
        b4 = np.load('b4')
    else:
        w1, b1 = init_weights(784, 256)
        w2, b2 = init_weights(256, 256)
        w3, b3 = init_weights(256, 256)
        w4, b4 = init_weights(256, 10)
    epochs = 50
    w1, w2, w3, w4, b1, b2, b3, b4, losses = minibatch_gd(50, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, 10)
    np.save('w1', w1)
    np.save('w2', w2)
    np.save('w3', w3)
    np.save('w4', w4)

    np.save('b1', b1)
    np.save('b2', b2)
    np.save('b3', b3)
    np.save('b4', b4)
    class_names = np.array(["T-shirt/top","Trouser","Pullover","Dress",
        "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])

    avg_class_rate, class_rate_per_class, y_pred = test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, 10)
    #plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                     # title='Confusion matrix, with normalization')
    print(avg_class_rate, class_rate_per_class)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    x = range(0, epochs)
    y = losses
    plt.plot(x,y)
    plt.show()
