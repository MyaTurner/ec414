function labels = assignDatapoints(dataMatrix, MU_init)

% Compute the euclidean distance for each row in given matrix from each
% given centroid

distanceFrom_mu = euclideanDistance(dataMatrix, MU_init);

% Find the minimum distance and assign the data point the appropiate label
% in labels vector
num_datapoints = length(distanceFrom_mu);
labels = zeros(num_datapoints,1);

% Finds minimum distance in each distance matrix row
minDistancefrom_mu = min(distanceFrom_mu,[],2);

for i = 1 : num_datapoints
    current_row = distanceFrom_mu(i, :);
    
    % Find column index, the label, to assign the data point
    labels(i) = find(current_row == minDistancefrom_mu(i));

end

