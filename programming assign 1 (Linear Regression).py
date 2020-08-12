# Importing packages
import numpy as np


# Q1
def gen_dataset(sigma, n, m):
    """ The function take in various parameters from the user including
    values of sigma, size of data set and the number of independent
    variables to return X, Y and beta."""

    # creating X numpy array
    np.random.seed(1)  # Fixing the seed for X
    X_orig = np.random.rand(n, m)
    # Adding column for 1's at the start of X
    X = np.hstack((np.ones((X_orig.shape[0], 1), dtype=X_orig.dtype), X_orig))

    # Defining beta
    np.random.seed(3)
    beta = np.random.rand(1, m+1)

    # creating e
    np.random.seed(4)
    e = np.random.normal(loc=0, scale=sigma)

    # print("########      X         #######")
    # print(X)
    # print("########      X_Orig        #######")
    # print(X_orig)
    # print(X.shape, beta.shape)

    # Calculating Y
    Y = np.dot(X, beta.T) + e

    return X_orig, Y, beta


# Q2
def linear_reg(X, Y, k, tao, learning_rate):
    """The function takes the input as X,Y,k and tao. It learns from the
    data given, the parameters using the gradient descent approach.

    X: An n x m numpy array of independent variable values
    Y : The n x 1 numpy array of output values
    k: the number of iterations (epochs)
    tao : the threshold on change in Cost function value from the previous to current iteration
    """

    # Getting shapes for defining Beta
    n = X.shape[0]
    m = X.shape[1]

    # Defining beta
    np.random.seed(1)
    beta = np.random.normal(size=(1, m+1))

    # Adding column of 1's
    X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
    # print(X)

    # Estimating Y_hat
    y_hat = np.dot(X, beta.T)

    # Calculating Error and Cost
    error = -(y_hat - Y)
    # print(error)
    cost = (np.sum(np.multiply(error, error), axis=0))/n
    # print("Epoch  0", "  - Cost: ", cost)

    # derivative = np.dot(2*error, X)

    # The Gradient Descent
    for i in range(1, k+1):
        # if i%10 == 0:
        # print("Epoch ", i, " - Cost: ", cost)

        # Calculating the gradient
        derivative = -2*np.dot(error.T, X)/n

        # New optimal value of Beta
        beta = beta - learning_rate*derivative

        # New reduced cost
        y_hat = np.dot(X, beta.T)
        error = -(y_hat - Y)
        current_cost = (np.sum(np.multiply(error, error), axis=0))/n

        # print(cost, current_cost, tao)

        # Checking the threshold
        if cost - current_cost <= tao:
            cost = current_cost
            break
        cost = current_cost

    return beta, cost, y_hat


# Uncomment following to test
'''while True:
    # getting parameters from user
    sigma = float(input("Enter value of Sigma for Unexplained Variation: "))
    n = int(input("Enter the size of data set: "))
    m = int(input("Enter the number of independent variables: "))

    # Generating Data
    X, Y, beta = gen_dataset(sigma, n, m)

    beta_trained, cost_trained, y_hat = linear_reg(X, Y, 1000, 0.0001, 0.1)

    # Checking Cosine similarity of the Beta's
    len_beta = np.sqrt(np.dot(beta, beta.T))
    len_beta_trained = np.sqrt(np.dot(beta_trained, beta_trained.T))
    cosine_sim = np.dot(beta, beta_trained.T)/(len_beta*len_beta_trained)

    # Comparing the Original Y and Beta with the trained ones
    print("######### Original Parameters (Beta) ###########")
    print(beta)
    print("######### Trained Parameters (Beta) ###########")
    print(beta_trained, "\n")
    print("######### Cosine similarity of  Parameters (Beta) ###########")
    print(cosine_sim, "\n")
    # print("######### Original Y ###########")
    # print(Y)
    # print("######### Trained Y ###########")
    # print(y_hat)'''
