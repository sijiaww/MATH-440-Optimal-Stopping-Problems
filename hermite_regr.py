import numpy as np
from numpy.polynomial.hermite import hermval
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


from typing import Callable

"""
S0 = 100      # Initial stock price
c = 10      
r = 0.05      # Risk-free rate
mu = 0.05     
sigma = 0.2  
T = 1  
"""

DEBUG = 1



np.array([
    [1, 1.09, 1.08, 1.34],
    [1, 1.16, 1.26, 1.54],
    [1, 1.22, 1.07, 1.03],
    [1, 0.93, 0.97, 0.92], 
    [1, 1.11, 1.56, 1.52],
    [1, 0.76, 0.77, 0.90], 
    [1, 0.92, 0.84, 1.01], 
    [1, 0.88, 1.22, 1.34]
])

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
def get_payout(curr_col : np.ndarray, value_equation : Callable[[np.float64], np.float64]) -> np.ndarray:
    cpy = np.array(curr_col)
    for i in range(cpy.shape[0]):
        cpy[i] = value_equation(cpy[i])

    return cpy

def hermite_features(x : np.ndarray, degree : int) -> np.ndarray: 
    return np.column_stack([hermval(x=x.flatten(), c= [0] * i + [1]) for i in range(degree + 1)])

# Returns the prediction from our hermite polynomial approximation of the conditional expectation 
def get_hermite_prediction(y_train : np.ndarray, x : np.ndarray, degree : int, value_equation : Callable[[np.float64], np.float64]):
    regr_model = make_pipeline(
        FunctionTransformer(lambda x: hermite_features(x=x, degree=degree)),
        LinearRegression()
    )
    
    #Only train the model on paths in the money 
    mask = np.array([value_equation(x[i, 0]) > 0 for i in range(x.shape[0])])
    in_the_money_x = x[mask].reshape((-1, 1))

    # If we have no in the money paths to train our regression on we just take regular expectation of the payouts in the future
    # Since we have no information to condition on 
    if len(in_the_money_x.flatten()) == 0: 
        return np.ones(y_train.shape) * np.mean(y_train), False
    
    in_the_money_y = y_train[mask].reshape((-1, 1))

    regr_model.fit(X=in_the_money_x, y=in_the_money_y)
    return regr_model.predict(X=x), True

def get_lin_reg_prediction(y_train : np.ndarray, x : np.ndarray, value_equation : Callable[[np.float64], np.float64]):
    mask = np.array([value_equation(x[i, 0]) > 0 for i in range(x.shape[0])])
    in_the_money_x = x[mask].reshape((-1, 1))

    if len(in_the_money_x.flatten()) == 0: 
        return np.ones(y_train.shape) * np.mean(y_train), False
    
    in_the_money_y = y_train[mask].reshape((-1, 1))

    regr_model = LinearRegression()
    regr_model.fit(X=in_the_money_x, y=in_the_money_y)

    return regr_model.predict(X=x), True 

def get_quad_regr_prediciton(y_train : np.ndarray, x : np.ndarray, degree : int,  value_equation : Callable[[np.float64], np.float64]):
    mask = np.array([value_equation(x[i, 0]) > 0 for i in range(x.shape[0])])
    in_the_money_x = x[mask].reshape((-1, 1))

    if len(in_the_money_x.flatten()) == 0: 
        return np.ones(y_train.shape) * np.mean(y_train), False
    
    in_the_money_y = y_train[mask].reshape((-1, 1))

    quad = PolynomialFeatures(degree=2, include_bias=False)
    transformed_x = quad.fit_transform(X=in_the_money_x)

    regr = LinearRegression()
    regr.fit(X=transformed_x, y=in_the_money_y)

    quad2 = PolynomialFeatures(degree=2, include_bias=False)

    x_quad = quad2.fit_transform(X=x)

    return regr.predict(X=x_quad), True


# Discounts a value
def discount_value(val : np.float64, r : np.float64, T : np.float64) -> np.float64:
    return np.exp(-1 * r * T) * val


def discount_col(col : np.ndarray, r, T) -> np.ndarray: 
    cpy = np.array(col)
    for i in range(cpy.shape[0]):
        cpy[i, 0] = discount_value(cpy[i, 0], r=r, T=T)

    return cpy


def main(S0 : int, num_paths : int, num_intervals : int, hermite_degree : int, sigma : np.float64, mu : np.float64, C : int, r : np.float64):
    basic_value_equation = lambda Sf: max(Sf - S0 - C, 0)
    put_value_equation = lambda Sf: max(S0 - Sf, 0)

    dt = 1

    #price_mat = generate_stock_prices(num_paths=num_paths, num_intervals= num_intervals, sigma=sigma, mu=mu, initial_price=S0, time_frame=1)
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
    print("PRICE MATRIX: \n")
    print(price_mat)
    print("\n")

    cash_flow_mat = np.zeros((num_paths, num_intervals))

    # Gets payout for time T
    cash_flow_mat[:, num_intervals - 1] = get_payout(curr_col=price_mat[:, num_intervals-1], value_equation=put_value_equation).flatten()

    print(cash_flow_mat)
    for t in range(num_intervals - 1, 0, -1):

        #price = get_payout(prices=price_mat, time=t, value_equation=put_value_equation).reshape((-1, 1))

        #Prices at time t
        price = cash_flow_mat[:, t].reshape((-1, 1))

        #Discount them by 1
        discounted_price = discount_col(price, r, dt)

        prev_price = price_mat[:, t-1].reshape((-1, 1))

        #predicted_payout, valid = get_quad_regr_prediciton(y_train=discounted_price, x=prev_price, degree=hermite_degree, value_equation=put_value_equation)
        #predicted_payout, valid = get_lin_reg_prediction(y_train=discounted_price, x=prev_price, value_equation=put_value_equation)
        predicted_payout, valid = get_hermite_prediction(y_train=discounted_price, x=prev_price, value_equation=put_value_equation, degree=hermite_degree)

        if not valid and t != 1:
            for p in range(num_paths):
                cash_flow_mat[p, t-1] = discount_value(cash_flow_mat[p, t], r, dt)

        # Gets predicted payout from waiting 
        predicted_payout = np.maximum(predicted_payout, 0).reshape((-1, 1))

        # Gets payout for exercising today 
        exercise_payout = get_payout(curr_col=prev_price, value_equation=put_value_equation).reshape((-1, 1))

        if DEBUG:
            print(f"AT TIME {t} PRICE: ")
            print(price)
            print(f"AT TIME {t - 1} PREDICTED FUTURE PAYOUT: ")
            print(predicted_payout)
            print("\n")
            print(f"AT TIME {t - 1} EXERCISE PAYOUT: ")
            print(exercise_payout)
            print("\n")

        # For each path 
        for p in range(num_paths):
            if exercise_payout[p].item() >= predicted_payout[p] and exercise_payout[p].item() != 0:
                cash_flow_mat[p, t-1] = exercise_payout[p].item()
                #Zero out future values 
                cash_flow_mat[p, t:] = 0
            else: 
                cash_flow_mat[p, t-1] = discount_value(cash_flow_mat[p, t], r, dt)
        if DEBUG: 
            print("CASH FLOW MATRIX: \n")
            print(cash_flow_mat)
            print("\n")

    print("CASH FLOW MATRIX: \n")
    print(cash_flow_mat)
    print("\n")

    valuation = np.mean(cash_flow_mat[:, 0], axis=0)
    print(max(valuation, put_value_equation(S0)))


    




if __name__ == '__main__':
    main(S0= 1.10, num_paths=8, num_intervals=4, mu=0.06, sigma=0.2, hermite_degree=2, C=1, r=0.06)