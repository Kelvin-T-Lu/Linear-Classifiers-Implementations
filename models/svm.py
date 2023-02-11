"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        return

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        # https://www.baeldung.com/cs/svm-multiclass-classification (Concept)
        # https://medium.com/@arsh1207/svm-implementation-from-scratch-in-python-8cf61a882ca8
        # https://ljvmiranda921.github.io/notebook/2017/02/11/multiclass-svm/
        # https://cs231n.github.io/optimization-1/
        # https://cs231n.github.io/linear-classify/
        # https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2 
        # https://github.com/qandeelabbassi/python-svm-sgd/blob/master/svm.py

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # ? Add bias function

        # Weights - (Num_Classes, D)
        # Rows - The weight vector for each class
        # Column - The weight w.r.t. feature columns.
        self.w = np.zeros((self.n_class, X_train.shape[1]))

        X_train = np.asarray(X_train)  # Converting to Cupy's NDArray

        for _ in range(self.epochs):
            for index, data_row in enumerate(X_train):
                # data_row = (1, D)
                # weights = (Num_Classes, D)
                y_pred = np.argmax(np.dot(data_row, self.w.T))
    
                # y_pred = 1
                y_correct = y_train[index]
                # print(f"Y_pred - {y_pred}, y_correct - {y_correct}")
                if y_pred != y_correct:  # Wrong prediction
                    # Update incorrect prediction w/ learnign rate.
                    self.w[y_pred] -= self.lr * data_row
                    # Update correct prediciton w/ learning rate.
                    self.w[y_correct] += self.lr * data_row

        pass

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
        return
