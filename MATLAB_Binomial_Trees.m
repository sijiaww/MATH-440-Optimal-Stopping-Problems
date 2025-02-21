%% code

% Parameters
S0 = 100; % Initial stock price
r = 0.05; % a constant for penalty
u = 1.003; % up
d = 0.997;  % down
pup = 0.49; % The probability of going up
c = 10; % service fee
T = 5; 

% Initializing the stock price and option value matrices
S = zeros(T+1, T+1);
V = zeros(T+1, T+1);

% Generate stock prices in binomial tree
for i = 0:T
    for j = 0:i
        S(j+1, i+1) = S0 * u^(i-j) * d^j;
    end
end

% Z matrix is the exercise value matrix
Z_matrix = zeros(T+1,T+1);
for i = 0:T
    for j = 0:i
        Z_matrix(j+1,i+1) = exp(-r*i)*(S(j+1,i+1)-c);
    end
end

% Set up the last column of V matrix
for j = 0:T
    V(j+1, T+1) = Z_matrix(j+1,T+1);
end

% Backward induction for V matrix
for i = T-1:-1:0
    for j = 0:i
        % Calculate the expected value of holding the option
        expectedValue = pup * V(j+1, i+2) + (1 - pup) * V(j+2, i+2);
        % Calculate the value of exercising the option now
        exerciseValue = Z_matrix(j+1,i+1);
        % The option value is the maximum of holding or exercising
        V(j+1, i+1) = max(expectedValue, exerciseValue);
    end
end

disp("The matrix S is :");
disp(S);
disp("The matrix V is :");
disp(V);
