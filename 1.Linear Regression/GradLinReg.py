class LinRegGradient:
    '''
    A light class for fitting Linear Regression using Gradient Descent algorithm

    Methods:
        get_cost(self, X, y, w, b)                      : Computes cost for linear regression
        get_gradient(self, X, y, w, b)                  : Computes gradient for linear regression
        fit(self, X, y, w_init, b_init, alpha, n_iter)  : Performs gradient descent for linear regression
        value_assert(self, X, y, alpha)
    '''


    def get_cost(self, X, y, w, b):
        '''
        Computes cost for linear regression

        Args:
         X: (ndarray(m,n)) : Feature values
         y: (ndarray(m,))  : Target values
         w: (ndarray(n,))  : Model parameters
         b: (scalar)       : Model parameter

        Returns:
         cost_total (scalar) : Linear regression cost
        '''

        import numpy as np

        m = X.shape[0]
        err = ((X.dot(w) + b) - y) ** 2
        cost = (err.sum() / (2 * m))

        return cost

    def get_gradient(self, X, y, w, b):

        '''
        Computes gradient for linear regression

        Args:
         X: (ndarray(m,n)) : Feature values
         y: (ndarray(m,))  : Target values
         w: (ndarray(n,))  : Model parameters
         b: (scalar)       : Model parameter

        Returns:
         dj_dw (ndarray(m,n)) : Gradient of the cost w.r.t parameters w
         dj_db (scalar)       : Gradient of the cost w.r.t parameter b
        '''

        import numpy as np

        m, n = X.shape

        err = (X.dot(w) + b) - y

        dj_db = err.sum()
        dj_dw = err.dot(X)

        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db

    def fit(self, X, y, w_init, b_init, alpha, n_iter):

        '''
        Performs gradient descent for linear regression

        Args:
         X (ndarray(m,n))      : Feature values
         y (ndarray(m,))       : Target values
         w_init (ndarray(n,)) : Initial model parameters
         b_init (scalar)      : Initial model parameter
         alpha                : Learning rate
         n_iter               : Number of iterations

        Returns:
         dj_dw (ndarray(m,n)) : Gradient of the cost w.r.t parameters w
         dj_db (scalar)       : Gradient of the cost w.r.t parameter b
        '''

        import numpy as np
        import math

        # Checking Arguments
        self.value_assert(X, y, w_init, b_init, alpha, n_iter)

        # Defining arrays for cost and parameters to be stored
        self.cost_arr = []
        self.w_arr = []
        self.b_arr = []

        # Initializing parameters
        self.w = w_init
        self.b = b_init

        for i in range(n_iter):

            # Calculating gradients
            dj_dw, dj_db = self.get_gradient(X, y, self.w, self.b)

            # Simultaneous update on both parameters
            self.w = self.w - alpha * dj_dw
            self.b = self.b - alpha * dj_db

            # Storing parameter and costs on each iteration
            if i < 50000:
                self.w_arr.append(self.w)
                self.b_arr.append(self.b)
                self.cost_arr.append(self.get_cost(X, y, self.w, self.b))

            # Printing each 1/10th iteration cost
            if i % math.ceil(n_iter / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.cost_arr[-1]:8.2f}  ")

        self.coef_ = [self.b, self.w]

    def value_assert(self, X, y, w_init, b_init, alpha, n_iter):

        '''
        Peforms input validation

         Args:
          X (ndarray(m,n))      : Feature values
          y (ndarray(m,))       : Target values
          w_init (ndarray(n,)) : Initial model parameters
          b_init (scalar)      : Initial model parameter
          alpha                : Learning rate
          n_iter               : Number of iterations

         Returns:
          None
        '''


        import numpy as np

        assert (np.isnan(X).sum() == 0) | (
                    np.isnan(y).sum() == 0), 'X feature matrix and or y target array can not contain null values'

        assert (X.dtype != 'str') & (
                    y.dtype != 'str'), 'X feature matrix and or y target array can not contain str values'

        assert isinstance(w_init[0],(int, float))  & isinstance(b_init, (int, float)), 'Initial model parameters w_init, b_init should be type of intereger or float'

        assert isinstance(alpha,(int, float)), 'Learning rate alpha should be type of intereger or float'

        assert isinstance(n_iter, int), 'Number of iterations n_iter should be type of intereger '

    def predict(self, X):

        '''
        Predicts outcome basde on fitted model parameters

        Args:
         X (ndarray(m,n)): Feature matrix of m observations

        Returns:
         preds (ndarray(m,)): Predicted outcomes
        '''

        import numpy as np

        preds = np.dot(X, self.w) + self.b

        return preds

    def plot_cost(self):

        '''
        PLots cost as a fucntion of iteration

         Args:
          None

        Returns:
         None
        '''

        import matplotlib.pyplot as plt
        import numpy as np

        plt.plot(np.arange(len(self.cost_arr)), self.cost_arr)
        plt.title('Cost vs Iteration')


