import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
We are going to give a dataframe to load_data which will split it into dependent and 
independent data and turn them into numpy array

y = np.reshape(y, (y.shape[0], 1)):

- y originally is a 1-dimensional array containing the labels for each sample.

- np.reshape(y, (y.shape[0], 1)) reshapes y from a 1-dimensional array of shape (m,) 
 (where m is the number of samples) to a 2-dimensional array of shape (m, 1).

- This reshaping ensures that y has the shape of a column vector, 
  which is necessary when performing matrix operations with self.__theta and x in logistic regression. 
  Without this, there would be alignment issues when calculating costs and gradients.

x = np.hstack((np.ones((x.shape[0], 1)), x)):

- Logistic regression requires an intercept term, often called the bias term, 
  which allows the model to fit data more flexibly by shifting the decision boundary.

- np.ones((x.shape[0], 1)) creates a column of ones with a length equal to the number of samples 
  in x. This column represents the intercept term.

- np.hstack((np.ones((x.shape[0], 1)), x)) horizontally stacks this column of ones with the 
  original x data. The new x has a shape of (m, n+1), where n is the original number of features (in this case, 2) and the extra column (of ones) is added as the first column.
  Adding this intercept column allows us to include the intercept in the model as a weight, simplifying the model so it can represent self.__theta as a single vector (with self.__theta[0] for the intercept and self.__theta[1:] for the feature weights).
"""


class CustomLogisticRegression:

    def __init__(
        self,
        learning_rate=0.001,
        num_epochs=1000,
        class_weights={0: 5, 1: 50},
        isBalanced=True,
        type = 'A'
    ):
        self.class_weights = class_weights
        self.learning_rate = learning_rate
        # self.tolerance = tolerance
        self.num_epochs = num_epochs
        self.isBalanced = isBalanced
        self.type = type
        self.__theta = None

    # This function takes a dataframe and lists of independent and dependent variables and convert them into matrices
    def load_data(self, df, ind, dep):
        i = df[ind]
        d = df[dep]
        x = np.array(i, dtype=float)
        y = np.array(d, dtype=float)
        y = np.reshape(y, (y.shape[0], 1))
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x, y

    # This the sigmnoid function for the logistic regression
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # This is the gradient decent without weights
    def gradient_descent(self, x, y):
        self.__theta = np.zeros((x.shape[1], 1))
        m = x.shape[0]
        j_all = []

        for _ in range(self.num_epochs):
            h_x = self.sigmoid(x @ self.__theta)
            cost_ = (1 / m) * (x.T @ (h_x - y))
            self.__theta = self.__theta - (self.learning_rate) * cost_
            j_all.append(self.cost_function(x, y, self.__theta))
        return self.__theta, j_all

    # This is the gradient decent with the weights
    def gradient_descent_with_class_weights(self, x, y, class_weights):
        self.__theta = np.zeros((x.shape[1], 1))
        m = x.shape[0]
        j_all = []

        for _ in range(self.num_epochs):
            # Compute predictions
            h_x = self.sigmoid(x @ self.__theta)

            # Calculate weighted error term
            weighted_errors = (h_x - y) * np.array(
                [class_weights[int(label)] for label in y.flatten()]
            ).reshape(-1, 1)

            # Compute the gradient using the weighted error term
            gradient = (1 / m) * (x.T @ weighted_errors)

            # Update self.__theta
            self.__theta = self.__theta - self.learning_rate * gradient

            # Track cost for analysis
            cost = self.cost_function(x, y, self.__theta)
            j_all.append(cost)

        return self.__theta, j_all

    def alternate_gradient_descent(self, x, y):
        self.__theta = np.zeros((x.shape[1], 1))
        m = x.shape[0]
        J_all = []

        high_risk_cost = 50  # Cost for predicting low risk when actual is high risk
        low_risk_cost = 5  # Cost for predicting high risk when actual is low risk

        for _ in range(self.num_epochs):
            h_x = self.sigmoid(x @ self.__theta)  # Predicted probabilities
            error = h_x - y  # Error term

            # Apply misclassification costs
            weighted_error = error * np.where(
                (y == 1) & (h_x < 0.5),
                high_risk_cost,
                np.where((y == 0) & (h_x >= 0.5), low_risk_cost, 1),
            )

            # Calculate the gradient with weighted error
            gradient = (1 / m) * (x.T @ weighted_error)
            self.__theta = self.__theta - self.learning_rate * gradient

            # Track cost without misclassification for reference
            J_all.append(self.cost_function(x, y, self.__theta))

        return self.__theta, J_all

    # This calculates the cost function
    def cost_function(self, x, y, theta):
        h = self.sigmoid(x @ theta)
        one = np.ones((y.shape[0], 1))
        cost = -((y.T @ np.log(h)) + (one - y).T @ np.log(one - h)) / (y.shape[0])
        return cost

    # Makes the prediction based on probabilities
    def classify(self, prob):
        if prob > 0.5:
            return 1
        else:
            return 0

    # Use this to test the model. Provide self.__theta and x_test to get perdicted y
    def predict(self, x):
        """Return predicted outputs for the given test data."""
        y_pred = np.array(
            [self.classify(self.sigmoid(x_i @ self.__theta)) for x_i in x]
        )
        return y_pred

    # This functions is used to train the model with training data. Pass x_train and y_train
    # Uses the isBalanced variable to decide which gradient decent to use
    def fit(self, x, y):
        if self.isBalanced:
            self.__theta, J_all = self.gradient_descent(x, y)
        else:
            if self.type=='A':
                self.__theta, J_all = self.gradient_descent_with_class_weights(
                    x, y, self.class_weights
                )
            elif self.type=='B':
                self.__theta, J_all = self.alternate_gradient_descent(x, y)
            # self.__theta, J_all = self.alternate_gradient_descent(x,y)
        J = self.cost_function(x, y, self.__theta)
        return self.__theta, J_all, J

    # Used to plot the cost vs iterations
    def plot_cost(self, J_all):
        n_epochs = []
        jplot = []
        count = 0
        for i in J_all:
            jplot.append(i[0][0])
            n_epochs.append(count)
            count += 1
        jplot = np.array(jplot)
        n_epochs = np.array(n_epochs)

        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.plot(n_epochs, jplot, "m", linewidth="2")
        plt.show()
