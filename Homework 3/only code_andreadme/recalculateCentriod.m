function newMU_init = recalculateCentriod(dataMatrix, labels, num_mu)

% For given labels, re-calculate the specified number of centriods, by
% by finding the mean of the x and y distance of all points for each label

% Initialize new MU
newMU_init = zeros(num_mu, 2);

for i = 1 : num_mu
    
    % Get current label value
    current_label = i;
    
    % Find all the data points with this given label
    currentLabel_data = dataMatrix((find(labels == current_label)), :);
    
    % If the current cluster has labels re-compute
    if(~isempty(currentLabel_data))
        
        % Calculate the mean x-distance for all the points with this given
        % label
        mu_x = mean(currentLabel_data(:,1));
    
        % Calculate the mean y-distance for all the points with this given
        % label
        mu_y = mean(currentLabel_data(:,2));
    
        % Add to new mu matrix
        newMU_init(i,:) = [mu_x mu_y];
    end
    
    
end


end

