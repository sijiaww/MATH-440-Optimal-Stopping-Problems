
% Parameters
K = 1.1;
num_paths = 15;
T = 1;
dt = 0.1;
S_0 = 1;
mu = 0.2;
sigma = 1;
r = 0.06;
% Kind of regression (e.g. 1 is linear, 2 is quadratic, etc.)
regression_degree = 2;

% The price matrix used in the paper - need to uncomment line 47 to use
test_matrix = [
 1 1.09 1.08 1.34; 
 1 1.16 1.26 1.54; 
 1 1.22 1.07 1.03;
 1 0.93 0.97 0.92;
 1 1.11 1.56 1.52;
 1 0.76 0.77 0.90;
 1 0.92 0.84 1.01;
 1 0.88 1.22 1.34;
 ]; 

% Currently set to the payoff function for a put - but you can define any
% payoff function here
function payoff = payoff_function(price)
    payoff = max(K - price, 0);
end

% First we generate the prices - similar to the binomial case

paths = zeros(num_paths, T/dt + 1); % Need the plus 1 for S_0 (b/c 1-indexing)
paths(:,1) = S_0;

for row = 1:num_paths
    for col = 2:(T/dt + 1) % First column is already done
        paths(row, col) = paths(row,col - 1) * exp((mu - (sigma^2 / 2)) * dt + sigma * sqrt(dt) * normrnd(0,1));
    end
end

% Paths are created at this point

% We construct the cash flow matrix (only the last column is relevant at this point)

% paths = test_matrix; num_paths = 8;% - UNCOMMENT TO DO THE TEST FROM THE PAPER

cash_flow_matrix = payoff_function(paths);



% Recursive Part:
dimensions = size(paths);
for col = (dimensions(2) - 1):-1:2 % Iterate over every column except the last and first

% Now we find the indices of the rows that are in the money 
itm_rows = find(cash_flow_matrix(:,col));

X = paths(itm_rows, col);

% Calculate Y matrix (complicated by the fact payoff could be multiple timesteps in the future)

Y = zeros(length(itm_rows),1);
for i = 1:length(itm_rows)
    row = itm_rows(i);

    future_cash_flow_location = find(cash_flow_matrix(row,col+1:end));

    if isempty(future_cash_flow_location)
        Y(i) = 0;
        continue
    end
    Y(i) = exp(-r * future_cash_flow_location) * cash_flow_matrix(row, col + future_cash_flow_location);
end


% Regression portion - Using polyfit here lets us do a general regression,
% but makes us require a for loop to get yCalc

b = polyfit(X,Y,regression_degree);

yCalc = zeros(length(X),1); % Initialize
for i = 1:(regression_degree + 1)
    yCalc = yCalc + b(i) * X .^ (regression_degree - i + 1);
end 
% The above essentially makes yCalc = b(1)X .^ deg + b(2) X .^ deg - 1 +
% ... + b(deg) * X .^ 0


for i = 1:length(itm_rows) % Decide whether to wait/continue at each itm row
    row = itm_rows(i);

    if cash_flow_matrix(row, col) >= yCalc(i)
        cash_flow_matrix(row, col+1:end) = 0;
    else
        cash_flow_matrix(row, col) = 0;
    end

end % End of ITM check
disp("iteration")
end % Big for statement



% Find value by discounting all cash flows to time 0 and averaging

running_total = 0;
for i = 1:num_paths
    cash_flow_location = find(cash_flow_matrix(i,2:end));

    if isempty(cash_flow_location) % Will never exercise in the future
    continue
    end

    running_total = running_total + exp(-r * cash_flow_location) * cash_flow_matrix(i,1 + cash_flow_location);
end

waiting_option_value = running_total / num_paths; % Simple average

% Determines whether we should exercise at t = 0 or hold past that
option_value = max(cash_flow_matrix(1,1), waiting_option_value);


% Print the cash flow matrix & option value in terminal
cash_flow_matrix
option_value

% Use this to graph the regression line against the actual points
%{
hold on
scatter(X(:,2), Y)
yCalc = b(1) + b(2) * X(:,2);
plot(X(:,2), yCalc)
grid on
%}