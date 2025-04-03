import numpy as np
from numpy.polynomial.hermite import hermval

from sklearn.linear_model import LinearRegression
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
def get_payout(prices : np.ndarray, time : int, value_equation : Callable[[np.float64], np.float64]) -> np.ndarray:
    curr_col = np.array(prices[:, time])
    for i in range(curr_col.shape[0]):
        curr_col[i] = value_equation(curr_col[i])

    return curr_col

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
        return np.ones(y_train.shape) * np.mean(y_train)
    
    in_the_money_y = y_train[mask].reshape((-1, 1))

    regr_model.fit(X=in_the_money_x, y=in_the_money_y)
    return regr_model.predict(X=x)

def get_lin_reg_prediction(y_train : np.ndarray, x : np.ndarray, value_equation : Callable[[np.float64], np.float64]):
    mask = np.array([value_equation(x[i, 0]) > 0 for i in range(x.shape[0])])
    in_the_money_x = x[mask].reshape((-1, 1))

    if len(in_the_money_x.flatten()) == 0: 
        return np.ones(y_train.shape) * np.mean(y_train)
    
    in_the_money_y = y_train[mask].reshape((-1, 1))

    regr_model = LinearRegression()
    regr_model.fit(X=in_the_money_x, y=in_the_money_y)

    return regr_model.predict(X=x)


# Discounts a value
def discount_value(val : np.float64, r : np.float64, T : np.float64) -> np.float64:
    return np.exp(-1 * r * T) * val


def main(S0 : int, num_paths : int, num_intervals : int, hermite_degree : int, sigma : np.float64, mu : np.float64, C : int, r : np.float64):
    basic_value_equation = lambda Sf: max(Sf - S0 - C, 0)

    dt = 1/num_intervals

    price_mat = generate_stock_prices(num_paths=num_paths, num_intervals= num_intervals, sigma=sigma, mu=mu, initial_price=S0, time_frame=1)
    print("PRICE MATRIX: \n")
    print(price_mat)
    print("\n")

    cash_flow_mat = np.zeros((num_paths, num_intervals))

    # Gets payout for time T
    cash_flow_mat[:, num_intervals - 1] = get_payout(prices=price_mat, time=num_intervals-1, value_equation=basic_value_equation).flatten()

    for t in range(num_intervals - 1, 0, -1):

        payout = get_payout(prices=price_mat, time=t, value_equation=basic_value_equation).reshape((-1, 1))
        prev_price = price_mat[:, t-1].reshape((-1, 1))

        # Gets predicted payout from waiting 
        predicted_payout = np.maximum(get_hermite_prediction(y_train=payout, x=prev_price, degree=hermite_degree, value_equation=basic_value_equation), 0).reshape((-1, 1))
        # Gets payout for exercising today 
        exercise_payout = get_payout(prices=price_mat, time=t-1, value_equation=basic_value_equation).reshape((-1, 1))

        if DEBUG:
            print(f"AT TIME {t - 1} PREDICTED FUTURE PAYOUT: ")
            print(predicted_payout)
            print("\n")
            print(f"AT TIME {t - 1} EXERCISE PAYOUT: ")
            print(exercise_payout)
            print("\n")

        # For each path 
        for p in range(num_paths):
            # Consider all of the values in our cash flow matrix from the future (should only be at most one non-zero entry per row) 
            # discount it to its value today 
            discounted_values = np.array([discount_value(val=cash_flow_mat[p, i], r=r, T=(dt * (i - t + 1)))  for i in range(t,num_intervals)])
            # Get the best payout we can recieve by waiting 
            max_future = np.max(discounted_values)

            # If the payout for exercising today is better than the max between our predicted payout for waiting and the maximum future payout 
            # Then we exercise today 
            if exercise_payout[p].item() >= max(discount_value(predicted_payout[p], r=r, T=dt), max_future):
                cash_flow_mat[p, t-1] = exercise_payout[p].item()
                #Zero out future values 
                cash_flow_mat[p, t:] = 0
        if DEBUG: 
            print("CASH FLOW MATRIX: \n")
            print(cash_flow_mat)
            print("\n")

    print("CASH FLOW MATRIX: \n")
    print(cash_flow_mat)
    print("\n")


    




if __name__ == '__main__':
    main(S0= 100, num_paths=10, num_intervals=5, mu=0.05, sigma=0.2, hermite_degree=3, C=1, r=0.05)