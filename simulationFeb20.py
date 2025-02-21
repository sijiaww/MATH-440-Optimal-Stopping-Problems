from statistics import mean
import numpy as np
from sklearn.linear_model import LinearRegression

S0 = 100      # Initial stock price
c = 10      
r = 0.05      # Risk-free rate
mu = 0.05     
sigma = 0.2  
T = 1  

def least_square( M, N):
    dt = T / M

    # Initialize the stock matrix 
    S = [[0] * (M + 1) for _ in range(N)]
    
    # Set initial stock price for all paths
    for i in range(N):
        S[i][0] = S0

    # Generate stock prices
    for i in range(N):
        for j in range(1, M + 1):
            Z = np.random.normal(0, 1)  # Zij iid normal(0,1)
            S[i][j] = S[i][j-1] * (np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z))

    # Initialize the cash flow matrix (payoff at maturity for each path)
    V = [[0] * (M+1) for _ in range(N)]

    for i in range(N):
        V[i][M] = (S[i][M] - c) * np.exp(-r * dt * M)

    # Initialize intercept and slope lists
    a_zero = [0] * M
    a_one = [0] * M

    # Backward induction for option pricing
    for j in range(M-1, -1, -1):
        
        variable = np.array([S[i][j] for i in range(N)]).reshape(-1, 1)
        argument = np.array([V[i][j+1] * np.exp(-r * dt) for i in range(N)])

        # Fit the linear model
        model = LinearRegression().fit(variable, argument)
        a_zero[j] = float(model.intercept_)
        a_one[j] = float(model.coef_[0])

        # Update values matrix
        for i in range(N):
            continue_value = a_zero[j] + a_one[j] * S[i][j]
            exercise_value = (S[i][j] - c) * np.exp(-r * dt * j)
            V[i][j] = float(max(exercise_value, continue_value))

    expected_v1 = mean([V[i][1] for i in range(N)])
    V0 = max((S0 - c), expected_v1)
    return V0, a_zero, a_one


M = 5         # Number of periods
N = 10        # Number of paths

V0, a_zero, a_one = least_square(M, N)
print(f"Initial option value V0: {f'{V0:.3f}'}")
print(f"Intercept coefficients: {[f'{x:.3f}' for x in a_zero]}")
print(f"Slope coefficients: {[f'{x:.3f}' for x in a_one]}")
