import random
from sklearn.datasets.samples_generator import make_regression 
import pylab as plt
from scipy import stats
import numpy as np

def gradient_descent_2(alpha, x, y, numIterations):
    m = x.shape[0] # number of samples
    n = x.shape[1] 
    theta = np.ones(n)
    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost
        # print "iter %s | J: %.3f" % (iter, J)      
        gradient = np.dot(x_transpose, loss) / m         
        theta = theta - alpha * gradient  # update
    return theta, J

def main_function(x_train,y_train, x_test, y_test, lr, num_iter):

    m, n = np.shape(x_train)
    x_train = np.c_[ np.ones(m), x_train] # insert column
    alpha = lr # learning rate
    theta, J = gradient_descent_2(alpha, x_train, y_train, num_iter)

    # print theta

    y_predict = []

    o, p = np.shape(x_test)
    x_test = np.c_[ np.ones(o), x_test]

    index = np.arange(x_test.shape[0])

    for i in range(x_test.shape[0]):
        y_predict.append(np.dot(x_test[i], theta))

    return y_predict


def main_function_with_plot(x_train,y_train, x_test, y_test, lr, num_iter, title):

    m, n = np.shape(x_train)
    x_train = np.c_[ np.ones(o), x_train] # insert column
    alpha = lr # learning rate
    theta, J = gradient_descent_2(alpha, x_train, y_train, num_iter)

    y_predict = []

    o, p = np.shape(x_test)
    x_test = np.c_[ np.ones(m), x_test]

    index = np.arange(x_test.shape[0])

    for i in range(x_test.shape[0]):
        y_predict.append(np.dot(x_test[i], theta))

    fig = plt.figure()
    plt.plot(index,y_train,'o', label = 'target_value')
    plt.plot(index,y_predict, marker = 's', color = 'red', label = 'function_value')
    plt.legend()
    plt.title(title + '_loss='+str(J))
    fig.savefig(title + '_lr=' + str(lr)+ '_num_iter='+ str(num_iter) + '.png')
    print "Done!"

# def predict_vectors(theta, x, )

def normalize(x_train, y_train, lr, num_iter):
    m, n = np.shape(x_train)
    x_train = np.c_[ np.ones(m), x_train] # insert column
    alpha = lr # learning rate
    theta, J = gradient_descent_2(alpha, x_train, y_train, num_iter)

    return theta, J