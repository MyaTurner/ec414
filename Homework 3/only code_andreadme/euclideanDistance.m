function distanceFrom_mu = euclideanDistance(dataMatrix, MU_init)

% Compute the euclidean distance for each row in given matrix from each
% given centroid

%Each row represents the data point and each column is its distance rom
% given centroid 

% Initializing distance vector
num_mu = height(MU_init);
num_datapoints = height(dataMatrix);
distanceFrom_mu = zeros(num_datapoints, num_mu);

for j = 1 : num_mu
    current_mu = MU_init(j, :);
    
    for i = 1 : num_datapoints
        distanceFrom_mu(i, j) = sqrt( sum ((dataMatrix(i, :) - current_mu).^2) );
    end
end
end

