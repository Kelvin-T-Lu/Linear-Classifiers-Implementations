"""Perceptron model."""

import numpy as np


from operator import add, sub
class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        # https://www.python-engineer.com/courses/mlfromscratch/06_perceptron/
        # https://medium.com/hackernoon/implementing-the-perceptron-algorithm-from-scratch-in-python-48be2d07b1c0

        # Multiclass perceptron
        # https://swayattadaw.medium.com/multiclass-perceptron-from-scratch-ed326fc34b8f
        # https://www.codingame.com/playgrounds/9487/deep-learning-from-scratch---theory-and-implementation/perceptrons
        # https://www.kaggle.com/code/alizahidraja/multiclass-perceptron
        # https://jermwatt.github.io/machine_learning_refined/notes/7_Linear_multiclass_classification/7_3_Perceptron.html
        # https://www.youtube.com/watch?v=EA627DC7k6M
        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        # Weights - (Num_Classes, D)
            # Rows - The weight vector for each class
            # Column - The weight w.r.t. feature columns. 
        self.w = np.random.rand(self.n_class, X_train.shape[1]) * 0.0001

        # TODO - Adding bias row. 
        bias = np.ones((self.n_class, 1))
        self.w = np.hstack((self.w,bias)) # Adding bias column.
        x_train_bias = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((X_train, x_train_bias))
        # X_train = np.asarray(X_train)  # Converting to Cupy's NDArray
        self.init_lr = self.lr
        # * Concept Updates
        #     if wrong:  
        #         w_pred -= class
        #         w_correct += class 

        for epoch_num in range(self.epochs):
            
            for index, data_row in enumerate(X_train):
                # data_row = (1, D)
                # weights = (Num_Classes, D)
                y_pred = np.argmax(np.dot(data_row, self.w.T))
                
                # y_pred = 1
                y_correct = y_train[index]
                # print(f"Y_pred - {y_pred}, y_correct - {y_correct}")
                if y_pred != y_correct: # Wrong prediction
                    self.w[y_pred] -= self.lr * data_row # Update incorrect prediction w/ learnign rate.
                    self.w[y_correct] += self.lr * data_row # Update correct prediciton w/ learning rate.
            
            # learning rate decay
            self.lr = (1/ (1 + epoch_num)) * self.init_lr
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

        # X_test - (N, D)
        # Weights - (num_classes, D)
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Adding bias vector
        X_test_weights = np.dot(X_test, self.w.T)

        y_pred = [np.argmax(data_row) for data_row in X_test_weights]

        
        return np.array(y_pred)
