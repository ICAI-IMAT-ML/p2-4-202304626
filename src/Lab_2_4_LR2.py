import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # X = np.c_[X,np.ones(X.shape[0])]  # añadimos una columna de 1s

        w = np.linalg.inv(X.T @ X) @ (X.T @ y)  # la @ se usa para hacer el producto, X.T representa la traspuesta

        self.intercept = w[0]  # extraemos el valor del término independiente (el primero del vector w)
        self.coefficients = w[1:]  # extraemos los coeficientes de la regresión (son todos los términos de w menos el primero)

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        # Para graficar en el notebook:
        self.loss_vals = []  
        self.params = []  

        # Implement gradient descent (TODO)
        for epoch in range(iterations):
            predictions = self.predict(X=X[:, 1:])  # X es una matriz con la primera columna de todo 1s, por lo que pasamos al preidict la segunda columna solo
            error = predictions - y   # en la diapo pone y-predictions
            # print(f"error: {error}")
            # print(f"error T: {error.T}")
            # print(f"X: {X}")

            # Calcular loss y almacenarla
            mse = np.mean(error**2)
            self.loss_vals.append(mse)

            # Valores actuales de intercept y coeficients
            self.params.append((self.intercept, self.coefficients))

            # TODO: Write the gradient values and the updates for the paramenters
            #gradient = (1/m)*np.sum(error[i]*X[i] for i in range(m))
            gradient = (1/m) * np.dot(error, X)  # dot es para hacer un producto vectorial (en este caso fila por matriz)

            """
            Lo planteamos como un producto de vector fila, error, (1xN) por matriz, X, (Nx2)
            De esta manera obtenemos un vector de 1x2, donde cada columna es lo que en la diapositiva
            sale como thetha sub j, es decir que para j=0 tenemos la parte del gradiente del intercept
            y para j = 1 tenemos la parte del gradiente de los coeficientes.

            Esto es equivalente a lo que nos dice la fórmula que para actualizar el thetha j, hay que hacer
            el sumatorio desde i=1 hasta i=N, de el error[i] multiplicado por el x[i] de la columna j.
            Este sumatorio es equivalente a hacer el producto vectorial de la fila error por la columna j de la matriz X, 
            porque se multiplican elemento a elemento y luego se suma todo. Así tenemos una forma más compacta.
            
            """

            #print(gradient)
            self.intercept -= learning_rate*gradient[0]
            self.coefficients -= learning_rate*gradient[1:]

            # TODO: Calculate and print the loss every 10 epochs
            if epoch % 100000 == 0:
                # mse = np.sum(np.power(y-predictions, 2)) / m 
                mse = np.power(evaluate_regression(y,predictions)["RMSE"],2)
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            predictions = X*self.coefficients + self.intercept  # Y = X*w + b  (modo unidimensional)
        else:
            predictions = X.dot(self.coefficients) + self.intercept  # Y = X*w + b  (modo multidimensional)
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    rss = np.sum((y_true-y_pred)**2)
    tss = np.sum((y_true-np.mean(y_true))**2)

    # R^2 Score
    r_squared = 1 - (rss/tss)

    # Root Mean Squared Error
    rmse = np.sqrt( np.sum(np.power(y_true-y_pred, 2)) /len(y_pred) )

    # Mean Absolute Error
    mae = (1/len(y_true))*np.sum(abs(y_true-y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        # TODO: Extract the categorical column
        categorical_column = X_transformed[:,index]

        # TODO: Find the unique categories (works with strings)
        unique_values = set(categorical_column)  # usando un set eliminamos los duplicados

        # TODO: Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = np.array([[1 if fila == value else 0 for value in unique_values] for fila in categorical_column])
        # pones 1s y 0s según corresponda, y eso lo hacemos por cada fila

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]  # esto sería para convertirlo en modo dummy encoding no?

        # TODO: Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        # Partimos la matriz en las columnas antes y después del índice donde queremos insertar nuestras nuevas columnas (one_hot)
        X_left = X_transformed[:, :index]  # Columnas antes de one_hot
        X_right = X_transformed[:, index + 1:]  # Columnas después de one_hot
        # Borramos la columna que había:
        X_transformed = np.delete(X_transformed, index, axis=1)
        # Concatenamos Izquierda + One-hot + Derecha usando hstack
        X_transformed = np.hstack((X_left, one_hot, X_right))
        # X_transformed = np.insert(X_transformed, index, one_hot, axis=1) # insertamos en X_transformed, las columnas one_hot (columnas por axis=1), en el índice index


    return X_transformed





##########################
##########################
##########################
##########################
##########################
##########################
##########################
#####################