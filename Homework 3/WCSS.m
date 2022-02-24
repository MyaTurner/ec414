function cost = WCSS(dataMatrix, labels, MU_init, num_mu)

% Get WCSS
cost = 0;

for i = 1 :num_mu
    
    % Get current label value
    current_label = i;
    
    % Get current MU x initialization
    currentMU_x = MU_init(i, 1);
    
    % Find all the data points with this given label
    currentLabel_data = dataMatrix((find(labels == current_label)), :);
    
    % Calculate cost
    diff = currentLabel_data(:, 1) - currentMU_x;
    cost = cost + sum( diff .^ 2);
   
end

end

