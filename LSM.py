import numpy as np
from numpy.polynomial.hermite import hermval
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


from typing import Callable

# Set DEBUG = 1 to print out verbose output, useful for debugging and also visualizing steps in algorithm 
DEBUG = 0

# Generates random stock prices for num_paths different paths of a stock over the course of num_intervals time intervals using a geometric brownian motion 
def generate_stock_prices(num_paths : int, num_intervals : int, sigma : np.float64, mu : np.float64, initial_price : np.float64, time_frame : int) -> np.ndarray:
    # Initialize our matrix and set the first column = S0
    prices_matrix = np.zeros((num_paths, num_intervals))
    prices_matrix[:, 0] = np.ones((num_paths, 1)).flatten() * initial_price 
    
    dt = time_frame / num_intervals

    for col in range(1, num_intervals):
        # Vector of values from N(0,1)
        Z = np.random.normal(loc=0, scale=1, size=(num_paths, 1))
        # Initialize each column in our matrix 
        prices_matrix[:, col] = prices_matrix[:, col - 1] * ((np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z))).flatten()
        if DEBUG: 
            print(f"Iter {col}")
            print(prices_matrix)

    return prices_matrix 

# Get the payout for selling at time t for each path 
# Essentially applies the value equation to an entire column and returns the resulting column 
def get_payout(curr_col : np.ndarray, value_equation : Callable[[np.float64], np.float64]) -> np.ndarray:
    cpy = np.array(curr_col)
    for i in range(cpy.shape[0]):
        cpy[i] = value_equation(cpy[i])

    return cpy

# Returns a matrix of our transformed x values using a hermite polynomial basis
def hermite_features(x : np.ndarray, degree : int) -> np.ndarray: 
    return np.column_stack([hermval(x=x.flatten(), c= [0] * i + [1]) for i in range(degree + 1)])

# Returns the prediction using hermite polynomials as a regression basis 
def get_hermite_prediction(y_train : np.ndarray, x : np.ndarray, degree : int, value_equation : Callable[[np.float64], np.float64]):
    
    # Creates pipeline to transform our x values using hermite polynomials and then applies a linear regression to the transformed values 
    regr_model = make_pipeline(
        FunctionTransformer(lambda x: hermite_features(x=x, degree=degree)),
        LinearRegression()
    )
    
    # Mask to get the values in our column which are currently in the money
    # We don't fit regression on values out of the money since for those paths we continue regardless of the what the regression says 
    mask = np.array([value_equation(x[i, 0]) > 0 for i in range(x.shape[0])])
    in_the_money_x = x[mask].reshape((-1, 1))

    # If we have no in the money paths to train our regression on we just take regular expectation of the payouts in the future
    # Since we have no information to condition on 
    if len(in_the_money_x.flatten()) == 0: 
        return np.ones(y_train.shape) * np.mean(y_train), False
    
    # Get the corresponding y values in the money 
    in_the_money_y = y_train[mask].reshape((-1, 1))

    # Fit our model to the in the money values and return the prediction 
    regr_model.fit(X=in_the_money_x, y=in_the_money_y)
    return regr_model.predict(X=x), True


# Same as get_hermite_prediction, but performs a standard linear regression instead
def get_lin_reg_prediction(y_train : np.ndarray, x : np.ndarray, value_equation : Callable[[np.float64], np.float64]):
    mask = np.array([value_equation(x[i, 0]) > 0 for i in range(x.shape[0])])
    in_the_money_x = x[mask].reshape((-1, 1))

    if len(in_the_money_x.flatten()) == 0: 
        return np.ones(y_train.shape) * np.mean(y_train), False
    
    in_the_money_y = y_train[mask].reshape((-1, 1))

    regr_model = LinearRegression()
    regr_model.fit(X=in_the_money_x, y=in_the_money_y)

    return regr_model.predict(X=x), True 

# Same as get_hermite_prediction but performs a polynomial regression of inputted degree instead 
def get_poly_regr_prediciton(y_train : np.ndarray, x : np.ndarray, degree : int,  value_equation : Callable[[np.float64], np.float64]):
    mask = np.array([value_equation(x[i, 0]) > 0 for i in range(x.shape[0])])
    in_the_money_x = x[mask].reshape((-1, 1))

    if len(in_the_money_x.flatten()) == 0: 
        return np.ones(y_train.shape) * np.mean(y_train), False
    
    in_the_money_y = y_train[mask].reshape((-1, 1))

    quad = PolynomialFeatures(degree=degree, include_bias=False)
    transformed_x = quad.fit_transform(X=in_the_money_x)

    regr = LinearRegression()
    regr.fit(X=transformed_x, y=in_the_money_y)

    quad2 = PolynomialFeatures(degree=degree, include_bias=False)

    x_quad = quad2.fit_transform(X=x)

    return regr.predict(X=x_quad), True


# Discounts a value
def discount_value(val : np.float64, r : np.float64, T : np.float64) -> np.float64:
    return np.exp(-1 * r * T) * val

# Calls discount value on an entire column
# Returns the discounted column 
def discount_col(col : np.ndarray, r, T) -> np.ndarray: 
    cpy = np.array(col)
    for i in range(cpy.shape[0]):
        cpy[i, 0] = discount_value(cpy[i, 0], r=r, T=T)

    return cpy


"""
S0: our assets initial starting price
num_paths: the number of paths we have in our price matrix (# rows in price matrix)
num_intervals: the number of time steps we have in our price matrix (# columns in our price matrix)
polynomial_degree: for hermite and polynomial basis regression is the degree of the polynomials used for the regression
sigma: our volitility for generating the price matrix (>0)
mu: our drift term for generating the price matrix (>=0)
r: the discounting rate 
dt: the length of our time intervals (should generally be 1/num_intervals)
value_euqation: a callable lambda function used for converting the price of our asset into our payout
prediction_type: Either "hermite", "polynomial" for hermite and polynomial predicitions. Otherwise model does linear regression for the prediction. 
price_mat: lets user initialize a price matrix, if None the price matrix generated according to generate_stock_prices

Function takes in above parameters and prints the Asset Price it calculates using Least Squared Monte-Carlo Algorithm 
"""
def main(S0 : int, num_paths : int, num_intervals : int, polynomial_degree : int, sigma : np.float64, mu : np.float64, r : np.float64, dt : np.float64, value_equation : Callable[[np.float64], np.float64], prediction_type : str, price_mat : np.ndarray = None):
    # If we haven't passed in a specific price matrix then generate one according to the inputted parameters 
    if not price_mat.any(): 
        price_mat = generate_stock_prices(num_paths=num_paths, num_intervals= num_intervals, sigma=sigma, mu=mu, initial_price=S0, time_frame=1)


    print("PRICE MATRIX: \n")
    print(price_mat)
    print("\n")

    # Initialize cash flow matrix
    cash_flow_mat = np.zeros((num_paths, num_intervals))
    # Gets payout for time T
    cash_flow_mat[:, num_intervals - 1] = get_payout(curr_col=price_mat[:, num_intervals-1], value_equation=value_equation).flatten()

    if DEBUG: 
        print(cash_flow_mat)

    # Start at last time step and work backwards 
    for t in range(num_intervals - 1, 0, -1):
        # Get the cash flows for the time t
        payout = cash_flow_mat[:, t].reshape((-1, 1))
        #Discount them by 1
        discounted_payout = discount_col(payout, r, dt)
        # Get the prices at time t-1
        prev_price = price_mat[:, t-1].reshape((-1, 1))

        # Get the predicted payout 
        if prediction_type == 'poly': 
            predicted_payout, valid = get_poly_regr_prediciton(y_train=discounted_payout, x=prev_price, degree=polynomial_degree, value_equation=value_equation)
        elif prediction_type == 'hermite':
            predicted_payout, valid = get_hermite_prediction(y_train=discounted_payout, x=prev_price, value_equation=value_equation, degree=polynomial_degree)
        else:
            predicted_payout, valid = get_lin_reg_prediction(y_train=discounted_payout, x=prev_price, value_equation=value_equation)
        
        # Check to make sure we were able to run the regression
        # If t-1==0 then we accept the predicted payout to just be an average of the future payouts
        # Otherwise every path was out of the money so we continue holding
        if not valid and t != 1:
            for p in range(num_paths):
                cash_flow_mat[p, t-1] = discount_value(cash_flow_mat[p, t], r, dt)

        # Gets predicted payout from waiting 
        predicted_payout = np.maximum(predicted_payout, 0).reshape((-1, 1))

        # Gets payout for exercising today 
        exercise_payout = get_payout(curr_col=prev_price, value_equation=value_equation).reshape((-1, 1))

        if DEBUG:
            print(f"AT TIME {t} PRICE: ")
            print(payout)
            print(f"AT TIME {t - 1} PREDICTED FUTURE PAYOUT: ")
            print(predicted_payout)
            print("\n")
            print(f"AT TIME {t - 1} EXERCISE PAYOUT: ")
            print(exercise_payout)
            print("\n")

        # For each path 
        for p in range(num_paths):
            # If excercising today is better than waiting, and if exercising today doesn't yield 0 value then we exercise today 
            if exercise_payout[p].item() >= predicted_payout[p] and exercise_payout[p].item() != 0:
                cash_flow_mat[p, t-1] = exercise_payout[p].item()
                #Zero out future values 
                cash_flow_mat[p, t:] = 0
            else: 
                # Otherwise we continue holding, so the optimal value for time t-1 is whatever the optimal value at time t is discounted back by a timestep 
                cash_flow_mat[p, t-1] = discount_value(cash_flow_mat[p, t], r, dt)
        if DEBUG: 
            print("CASH FLOW MATRIX: \n")
            print(cash_flow_mat)
            print("\n")

    
    print("CASH FLOW MATRIX: \n")
    print(cash_flow_mat)
    print("\n")

    # column 0 in our cash flow matrix holds all of the optimal payouts discounted back to our start time
    # average them and compare with the payout from exercising at time 0 
    # return the best option 
    valuation = np.mean(cash_flow_mat[:, 0], axis=0)
    print(f"Calculated Price: {max(valuation, value_equation(S0))}")


    




if __name__ == '__main__':
    # Some value equations to use in model 
    basic_value_equation = lambda Sf: max(Sf - 1.10 - 1, 0)
    put_value_equation = lambda Sf: max(1.10 - Sf, 0)

    # price matrix for paper example 
    price_mat = np.array([
        [1, 1.09, 1.08, 1.34],
        [1, 1.16, 1.26, 1.54],
        [1, 1.22, 1.07, 1.03],
        [1, 0.93, 0.97, 0.92], 
        [1, 1.11, 1.56, 1.52],
        [1, 0.76, 0.77, 0.90], 
        [1, 0.92, 0.84, 1.01], 
        [1, 0.88, 1.22, 1.34]
    ])  


    main(S0= 1.10, num_paths=8, num_intervals=4, mu=0.06, sigma=0.2, polynomial_degree=2, r=0.06, dt=1, value_equation=put_value_equation, prediction_type='poly', price_mat=price_mat)