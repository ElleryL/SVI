

import data
import autograd.numpy as np

from scipy.special import logsumexp
import matplotlib.pyplot as plt
np.random.seed(14)
import autograd.scipy.stats.norm as norm
N_data, train_images, train_labels, test_images, test_labels = data.load_mnist()
D = 784
C = 10

def binarize_data(input_data):
    return np.where(input_data>0.5, 1.0,0.0)

def log_softmax(X, weight):
    '''
    :param: train_images:= NxD
    :param weight: DxC matrix, where D:= dimension of data; C:= num of classes
    :return: NxC matrix. Each row is probability dist for corresponding data
    '''
    z = np.dot(X, weight)
    deno = logsumexp(z,axis=1)
    return ((z-deno[:,np.newaxis]))

def log_likelihood(labels, z, weight, sigma):

    likelihood = np.mean(np.multiply(labels, (z)))
    reg = np.sum(-1/2*np.log(2*np.pi*sigma) - weight**2/(2*sigma))
    return likelihood+reg


def gradient(X, labels,z,weight,sigma):
    '''
    :param: z is log softmax NxC.
    :return: gradient DxC
    '''

    return -(np.dot((np.exp(z)-labels).T, X).T)/X.shape[0] - weight/(sigma)/300



def optimization(alpha,batch_images,batch_labels,sigma):

    weight = np.zeros((D,C))
    loss = []
    iters = 0
    z = log_softmax(batch_images, weight)
    loss_per_iter = log_likelihood(batch_labels, z, weight, sigma)
    loss.append(loss_per_iter)

    while (iters<500):

        g = gradient(batch_images,batch_labels,z,weight,sigma)
        weight = weight + alpha*g
        z = log_softmax(batch_images, weight)
        loss_per_iter = log_likelihood(batch_labels, z, weight, sigma)
        loss.append(loss_per_iter)
        if iters%200 == 0:
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
    test_class_label = np.argmax(test_labels, axis=1)
    diff_sigma = [1,10,50,100]
    accu = []

    best_weight = None
    highest_likeli = float('-inf')
    best_sig = 0
    for s in diff_sigma:

        weight,loss = optimization(0.1,train_images[:300],train_labels[:300],s)
        z = log_softmax(test_images, weight)
        likelihood = log_likelihood(test_labels, z, weight, s) # compute test likelihood for sigma

        # update the highest likelihood sigma
        if (likelihood) > highest_likeli:

            highest_likeli = (likelihood)

            best_weight = weight
            best_sig = s

    print("highest likelihood sigma is {}".format(best_sig))
    data.save_images(best_weight.T,"Q3_weight")
    data.plot_images(best_weight.T, plt.figure().add_subplot(111))
    #plt.close()



    print("Train Set Result:")
    class_label = np.argmax(train_labels[:300], axis=1)
    pred, avg = prediction(train_images[:300], best_weight, class_label)
    print("avg prediction train accuracy is {}".format(np.mean(pred == class_label)))
    print("The predictive avg log-likelihood is {}".format(avg))

    print("\n")

    print("Test Set Result:")

    pred, avg = prediction(test_images, best_weight, test_class_label)
    print("avg prediction test accuracy is {}".format(np.mean(pred == test_class_label)))
    print("The predictive avg log-likelihood is {}".format(avg))

    plt.plot(accu)
    plt.show()

