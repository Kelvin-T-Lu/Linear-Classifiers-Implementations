"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        W = self.w.T
        reg = self.reg_const
        X = X_train
        y = y_train
        loss = 0
        dW = np.zeros_like(W)

        num_classes = W.shape[1]
        num_train = X.shape[0]
        for i in range(num_train):
            scores = X[i].dot(W) # scores.shape is N x C

            # shift values for 'scores' for numeric reasons (over-flow cautious)
            scores -= scores.max()

            probs = np.exp(scores)/np.sum(np.exp(scores))

            loss += -np.log(probs[y[i]])

            # since dL(i)/df(k) = p(k) - 1 (if k = y[i]), where f is a vector of scores for the given example
            # i is the training sample and k is the class
            dscores = probs.reshape(1,-1)
            dscores[:, y[i]] -= 1

            # since scores = X.dot(W), iget dW by multiplying X.T and dscores
            # W is D x C so dW should also match those dimensions
            # X.T x dscores = (D x 1) x (1 x C) = D x C
            dW += np.dot(X[i].T.reshape(X[i].shape[0], 1), dscores)

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        loss /= num_train
        dW /= num_train

        # Add regularization to the loss.
        loss += reg * np.sum(W * W)

        # Add regularization loss to the gradient
        dW += 2 * reg * W    

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return dW
        # return

    def train(self, X_train: np.ndarray, y_train: np.ndarray, batch_size = 150):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # Weights - (Num_Classes, D)
        # Rows - The weight vector for each class
        # Column - The weight w.r.t. feature columns.
        self.w = np.random.rand(self.n_class, X_train.shape[1]) * 0.0001

        # Add bias row.
        bias = np.ones((self.n_class, 1))
        self.w = np.hstack((self.w, bias))  # Adding bias column.
        x_train_bias = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((X_train, x_train_bias))
        # X_train = np.asarray(X_train)  # Converting to Cupy's NDArray

        # Gradient descent with batches. 
        for i in range(self.epochs):
            mask_indicies = np.random.choice(X_train.shape[0], batch_size, replace = True)

            X_batch, y_batch = X_train[mask_indicies], y_train[mask_indicies]

            gradient = self.calc_gradient(X_batch, y_batch).T

            self.w -= self.lr * gradient

            print(i)
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

        X_test_weights = np.dot(X_test, self.w.T)

        y_pred = [np.argmax(data_row) for data_row in X_test_weights]

        return y_pred
