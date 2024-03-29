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
        self.w = None  
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
        W = self.w.T # (D, n_classes)
        cross_entropy_loss = 0
        
        dW = np.zeros_like(W)

        for row_index in range(X_train.shape[0]):
            
            # (N x D) * (D x N_Classes)
            score_vectors = X_train[row_index].dot(W) #  N x N_Classes

            score_vectors -= score_vectors.max() # Prevent overflow

            # Cross entropy calculation
            class_probs = np.exp(score_vectors)/np.sum(np.exp(score_vectors))

            cross_entropy_loss += -np.log(class_probs[y_train[row_index]])

            # Reshaping class scores into columns.
            class_scores = class_probs.reshape(1,-1)
            class_scores[:, y_train[row_index]] -= 1

            # (D x 1) x (1 x N_classes) = D x n_classes
            dW += np.dot(X_train[row_index].T.reshape(X_train[row_index].shape[0], 1), class_scores)

        # 1/N, averages of loss and gradient. 
        cross_entropy_loss /= X_train.shape[0]
        dW /= X_train.shape[0]

        # Regularization. 
        cross_entropy_loss += self.reg_const * np.sum(np.square(W))

        # Add regularization loss for the gradient
        dW += self.reg_const * W    


        return dW, cross_entropy_loss

    def train(self, X_train: np.ndarray, y_train: np.ndarray, batch_size = 150, verbose = False):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels.
            batch_size(int): Size of batches for batch SGD.
        """

        # Weights - (Num_Classes, D)
        # Rows - The weight vector for each class
        # Column - The weight w.r.t. feature columns.
        self.w = np.random.rand(self.n_class, X_train.shape[1]) * 0.0005

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

            gradient, loss = self.calc_gradient(X_batch, y_batch)

            self.w -= self.lr * gradient.T

            if verbose: # If verbose = True, print the loss. 
                print(f"Epoch - {i}, Cross Entropy Loss {loss}")

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
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Adding Bias vector

        # Class classification.
        X_test_weights = np.dot(X_test, self.w.T)

        y_pred = [np.argmax(data_row) for data_row in X_test_weights]

        return y_pred
