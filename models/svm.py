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
        self.w = None 
        self.lr = lr
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

        delta = 1 # Delta from CS231N Linear Classify
        W = self.w.T # (Features, Class_label)

        dW = np.zeros(W.shape) # Intial gradient.

        # compute the loss and the gradient
        hinge_loss = 0.0
        
        for row_index in range(X_train.shape[0]): # For each instance inside batch

            scores = X_train[row_index].dot(W) # Calculated scores for class. 
            correct_c_vector = scores[y_train[row_index]] # Scores for correct class

            for class_label in range(self.n_class):
                if class_label == y_train[row_index]: # Classes match, continue. 
                    continue
                # max(0, class_score - predicted_scores + delta)
                class_loss = scores[class_label] - correct_c_vector + delta # note delta = 1
                if class_loss > 0:
                    hinge_loss += class_loss

                    # Gradiant w.r.t. to c != yi
                    dW[:, class_label] += X_train[row_index].T
                
                    # Gradiant w.r.t. Wyi (Correct class)
                    dW[:, y_train[row_index]] += -X_train[row_index].T
            
        # 1/N, for averages. 
        hinge_loss /= X_train.shape[0]
        dW /= X_train.shape[0]

        # Regularization portion of equation (lambda * ||W||^2)
        hinge_loss += self.reg_const * np.sum(np.square(W))
        
        # Add regularization loss to the gradient
        dW += self.reg_const * W

        return dW, hinge_loss

    def train(self, X_train: np.ndarray, y_train: np.ndarray, batch_size = 150, loss_verbose = False):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        # https://www.baeldung.com/cs/svm-multiclass-classification (Concept)
        # https://medium.com/@arsh1207/svm-implementation-from-scratch-in-python-8cf61a882ca8
        # https://ljvmiranda921.github.io/notebook/2017/02/11/multiclass-svm/
        # https://cs231n.github.io/optimization-1/
        # https://cs231n.github.io/linear-classify/
        # https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2 
        # https://github.com/qandeelabbassi/python-svm-sgd/blob/master/svm.py
        # https://github.com/amanchadha/stanford-cs231n-assignments-2020/tree/master/assignment1/cs231n/classifiers
        # https://github.com/ibayramli/Multiclass-SVM-Image-Classifier
        # https://github.com/HuangYukun/columbia_cs_deep_learning_1/blob/master/ecbm4040/classifiers/linear_svm.py
        # https://users.cs.utah.edu/~zhe/pdf/lec-19-2-svm-sgd-upload.pdf
        

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
            batch_size(int): Size of batches for gradient descent calculation. 
            loss_verbose(bool): Set to true to print hinge_loss for each epoch iteration.
        """
        # self.w - (Num_Classes, Feature_Cols)
            # Rows - The weight vector for each class
            # Column - The weight w.r.t. feature columns.
        self.w = np.random.rand(self.n_class, X_train.shape[1]) * 0.0005

        # Add bias row.
        bias = np.ones((self.n_class, 1))
        self.w = np.hstack((self.w, bias))  # Adding bias column.
        x_train_bias = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((X_train, x_train_bias))

        # Gradient descent with batches. 
        for i in range(self.epochs):

            # Selecting batches. 
            mask_indicies = np.random.choice(X_train.shape[0], batch_size, replace = True)
            X_batch, y_batch = X_train[mask_indicies], y_train[mask_indicies]

            # Gradiant descent. 
            gradient, loss = self.calc_gradient(X_batch, y_batch)
            self.w -= self.lr * gradient.T 

            # Option to report hinge loss.
            if loss_verbose:  
                print(f"Epoch {i} Loss - {loss}")


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
        # Adding bias vector
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

        # Predictions
        X_test_weights = np.dot(X_test, self.w.T)

        y_pred = [np.argmax(data_row) for data_row in X_test_weights]

        return np.array(y_pred)
