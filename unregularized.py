import data
import autograd.numpy as np

from scipy.special import logsumexp
from Q1_reg import binarize_data
import matplotlib.pyplot as plt
np.random.seed(14)


N_data, train_images, train_labels, test_images, test_labels = data.load_mnist()
D = 784
C = 10

def log_softmax(X, weight):
    '''
    :param: train_images:= NxD
    :param weight: DxC matrix, where D:= dimension of data; C:= num of classes
    :return: NxC matrix. Each row is probability dist for corresponding data
    '''
    z = np.dot(X, weight)
    deno = logsumexp(z,axis=1)
    return ((z-deno[:,np.newaxis]))

def neg_log_likelihood(labels,z):

    loss = np.mean(np.multiply(labels, (z)))
    return -loss


def gradient(X, labels,z,weight):
    '''
    :param: z is log softmax NxC.
    :return: gradient DxC
    '''

    return (np.dot((np.exp(z)-labels).T, X).T)/X.shape[0]



def optimization(alpha,batch_images,batch_labels):

    weight = np.zeros((D,C))
    loss = []
    iters = 0
    z = log_softmax(batch_images, weight)
    loss_per_iter = neg_log_likelihood(batch_labels, z)
    loss.append(loss_per_iter)

    while (iters<500):

        g = gradient(batch_images,batch_labels,z,weight)
        weight = weight - alpha*g
        z = log_softmax(batch_images, weight)
        loss_per_iter = neg_log_likelihood(batch_labels, z)
        loss.append(loss_per_iter)
        if iters%100 == 0:
            print("{}th iteration has loss value {}".format(iters, loss[-1]))
        iters += 1
    print("Finish the train with {} iteration, loss value {}, alpha={}".format(iters,loss[-1],alpha))
    return weight,loss


def avg_log_likelihood(prob,Y):

    result = np.mean(prob[np.arange(prob.shape[0]), Y])
    return (result)

def prediction(X,weight,Y):
    log_prob = log_softmax(X, weight)
    avg = avg_log_likelihood(((log_prob)),Y)
    pred = np.argmax(np.exp(log_prob),axis=1)
    return pred, avg



if __name__ == "__main__":

    train_images = binarize_data(train_images)
    test_images = binarize_data(test_images)
    testY = np.argmax(test_labels, axis=1)
    trainY = np.argmax(train_labels, axis=1)
    weight,loss = optimization(0.1,train_images[:300],train_labels[:300])

    data.save_images(weight.T,"unreg_Q3_weight")
    data.plot_images(weight.T, plt.figure().add_subplot(111))
    plt.show()

    print("Train Set Result:")
    class_label = np.argmax(train_labels[:300], axis=1)
    pred, avg = prediction(train_images[:300], weight, class_label)
    print("avg prediction Error is {}".format(np.mean(pred != class_label)))
    print("The predictive avg log-likelihood is {}".format(avg))

    print("\n")

    print("Test Set Result:")
    class_label = np.argmax(test_labels,axis=1)
    pred, avg = prediction(test_images, weight, class_label)
    print("avg prediction Error is {}".format(np.mean(pred != class_label)))
    print("The predictive avg log-likelihood is {}".format(avg))
