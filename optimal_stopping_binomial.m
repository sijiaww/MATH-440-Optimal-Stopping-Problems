function value_tree = optimal_stopping_binomial(n, S_0, r, c, p_u, p_d, u, d)
% Does not work for S_0 = 0 (pretty self explanatory)
if p_u + p_d ~= 1 
    return
end

price_tree = zeros(n+1,n+1);

price_tree(1,1) = S_0; 

for i = 1:n
    new_col = zeros(1,n+1);
    counter = 2;
    new_col(1) = u * price_tree(1,i);
    for j = 1:n+1
        if price_tree(j,i) == 0
            break
        end
        new_col(counter) =  d * price_tree(j,i);
        counter = counter + 1;
    end
    price_tree(:,i+1) = new_col';
end

% Now we need to calculate the value matrix

value_tree = price_tree;
decision_tree = zeros(n+1,n+1);

value_tree(:,n+1) = (value_tree(:,n+1) - c) * exp(-n*r); % Discount last column
decision_tree(:,n+1) = decision_tree(:,n+1) + 2;

for i = n:-1:1
    for j = 1:n+1
    if value_tree(j,i) == 0
        break
    end
       current_payout = exp(-(i-1)*r) * (value_tree(j,i) - c); % i - 1 because we're indexing S_0 as 1 (matlab)
       wait_expected_payout = p_u * value_tree(j,i+1) + p_d * value_tree(j+1,i+1);
       % max(current_payout, wait_expected_payout);
       if current_payout > wait_expected_payout % this if,else can be written as just max(_,_), but I wanted to make a decision tree 
           value_tree(j,i) = current_payout;
           decision_tree(j,i) = 2;
       else
           value_tree(j,i) = wait_expected_payout;
           decision_tree(j,i) = 1;
       end
       
    end
end

price_tree

decision_tree % 2 is sell, 1 is wait
