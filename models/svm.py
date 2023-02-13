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
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        # TODO - Refactor me

        reg = self.reg_const
        W = self.w.T

        dW = np.zeros(W.shape) # initialize the gradient as zero

        # compute the loss and the gradient
        num_classes = W.shape[1]
        num_train = X.shape[0]
        loss = 0.0
        for i in range(num_train):
            scores = X[i].dot(W)
            correct_class_score = scores[y[i]]
            for j in range(num_classes):
                if j == y[i]: # Classes match
                    continue
                margin = scores[j] - correct_class_score + 1 # note delta = 1
                if margin > 0:
                    loss += margin

                    # for incorrect classes (j != y[i]), gradient for class j is x * I(margin > 0) 
                    # the transpose on the extracted input sample X[i] transforms it into a column vector
                    # for dw[:, j]
                    dW[:, j] += X[i].T
                
                    # for correct class (j = y[i]), gradient for class j is -x * I(margin > 0) 
                    # the transpose on the extracted input sample X[i] transforms it into a column vector
                    # for dw[:, j]
                    dW[:, y[i]] += -X[i].T
            
        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        loss /= num_train
        dW /= num_train

        # Add regularization to the loss.
        loss += reg * np.sum(W * W)

        # Add regularization loss to the gradient
        dW += 2 * reg * W

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
        # return loss, dW
        return dW
        # print(scores.shape, correct_y_vectors.shape)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, batch_size = 150):
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
        # Adding bias vector
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

        X_test_weights = np.dot(X_test, self.w.T)

        y_pred = [np.argmax(data_row) for data_row in X_test_weights]

        return np.array(y_pred)
