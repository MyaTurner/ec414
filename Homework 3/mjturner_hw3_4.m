% EC 414 - HW 3 - Spring 2022
% DP-Means starter code

clear, clc, close all,

%% 3.4b Generate Gaussian data:
% Intializing mean vectors
mu1 = [2, 2]';
mu2 = [-2,  2]';
mu3 = [0, -3.25]';

% Intializing covariance vectors
identityMatrix = [1, 0; 0, 1];
sigma1 = 0.02 .* identityMatrix;
sigma2 = 0.05 .* identityMatrix;
sigma3 = 0.07 .* identityMatrix;

% Creating Gaussian Data Clusters
gaussian1 = mvnrnd(mu1,sigma1,50);
gaussian2 = mvnrnd(mu2,sigma2,50);
gaussian3 = mvnrnd(mu3,sigma3,50);

DATA = [gaussian1; gaussian2;  gaussian3];

% Parameter Initializations
LAMBDA = 0.15;
convergence_threshold = 1;
MU = [];
MU = [MU; mean(DATA,1)];
x_name = 'First Feature';
y_name = 'Second Feature';

DPmeans(DATA, LAMBDA, MU, convergence_threshold, x_name, y_name);

%% 3.4c Generate NBA data:
NBA = readmatrix("NBA_stats_2018_2019.xlsx");
points = NBA(:,7);
minutes = NBA(:,5);
% DATA = [minutes points];

%% Functions
function DPmeans(dataMatrix, lambda, mu, convergence_threshold, x_name, y_name)

% Initializations
k = 1;
converged = 0;
mu_array = [];
mu_array = [mu_array; mu];
t = 0;
clusters = {};
clusters = [clusters [1 : height(dataMatrix)] ];

% Iterator for mu
current_MU = mu;

%Iterator for mu_array
currentMU_array = mu_array;

% Initialze all labels of data matrix to 1.  Each row corresponds to a data
% point
labels = ones(length(dataMatrix), 1);

while(converged == 0)
    
    t = t + 1;
    fprintf('Current iteration: %d...\n',t)
    
    %Iterators
    prev_k = k;
    
    for i = 1 : height(dataMatrix)
        
        % Get Current point
        current_point = dataMatrix(i, :);
        
        % Find distance for each point from its label center mu
        distanceFrom_mu = euclideanDistance(dataMatrix, current_MU);
    
        % Find the point with the minimum distance from above
        minDistance = min(distanceFrom_mu);
        min_indices = find(distanceFrom_mu == minDistance);
        
        % Define and do penalty check
        penalty = minDistance ^ 2;
        if(penalty > lambda)
            k = k + 1;
            labels(min_indices) = k;
            currentMU_array = [currentMU_array; current_point];
            current_MU = current_point;
            
        end
    end
    
    % Find the indices for all points with the same label k to form
    % clusters
    for label = 1 : k
        clusters{label} = find(labels == label);
    end
    
    % Compute a new means for each cluster (number of clusters is equal to k)
    newMU_array = zeros(k, 2);
    
    for i = 1 : k
        newMU_array(i, :) = mean( dataMatrix(clusters{i},:), 1 );
    end
    
    % Check for convergence by checking if k is not changing and the
    % cluster means by a specifed threshold
    convergenceMetric = abs( sum( sum (currentMU_array - newMU_array) ) );
    if(convergenceMetric <= convergence_threshold && prev_k == k)
        
        converged = 1;
    else
        currentMU_array = newMU_array;
    end
    
    if(converged)
        
        % Plot final clusters after convergence
         figure
        % Getting points for each label
        for i = 1 : k
            % Getting points for each label
            labeledData = dataMatrix((find(labels == i)), :);
            
            % Create Scatter Plot
            
            hold on
            scatter(labeledData(:,1), labeledData(:, 2));
            
            % label axis and title
            xlabel(sprintf('%s', x_name))
            ylabel(sprintf('%s', y_name))
            title (sprintf('Clusters for lambda = %.2f ',lambda))
        end
    end
    
end
end

%% DP Means method:

% Parameter Initializations
% LAMBDA = 0.15;
% convergence_threshold = 1;
% num_points = length(DATA);
% total_indices = [1:num_points];
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% DP Means - Initializations for algorithm %%%
% % cluster count
% %K = 1;
% 
% % sets of points that make up clusters
% L = {};
% L = [L [1:num_points]];
% 
% % Class indicators/labels
% Z = ones(1,num_points);
% 
% % means
% MU = [];
% MU = [MU; mean(DATA,1)];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% [K, mu_array] = DPmeans(DATA, lambda, MU, convergence_threshold);
% 
% % Initializations for algorithm:
% % converged = 0;
% % t = 0;
% % while (converged == 0)
% %     t = t + 1;
% %     fprintf('Current iteration: %d...\n',t)
%     
%     %% Per Data Point:
%     for i = 1:num_points
%         
%         %% CODE 1 - Calculate distance from current point to all currently existing clusters
%         distanceFrom_mu = euclideanDistance(currentLabel_data, mu);
%         
%         %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
%         % Write code below here:
% 
%     end
%     
%     %% CODE 3 - Form new sets of points (clusters)
%     % Write code below here:
%     
%     %% CODE 4 - Recompute means per cluster
%     % Write code below here:
%     
%     %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
%     % Write code below here:
%     
%     if (converged)
%          %% CODE 6 - Plot final clusters after convergence 
%          % Write code below here:
%     end    
% end