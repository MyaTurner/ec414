function labels = assignDatapoints(dataMatrix, MU_init)

% Compute the euclidean distance for each row in given matrix from each
% given centroid
distanceFrom_mu = euclideanDistance(dataMatrix, MU_init);

% Finds minimum distance in each row and find column index
[~, labels] = min(distanceFrom_mu');

% Return labels
labels = labels';