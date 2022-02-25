% EC 414 - HW 3 - Spring 2022
% DP-Means starter code

clear, clc, close all,

%% Generate Gaussian data:
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


%% Generate NBA data:
NBA = readmatrix("NBA_stats_2018_2019.xlsx");
points = NBA(:,7);
minutes = NBA(:,5);
% DATA = [minutes points];
%% DP Means method:

% Parameter Initializations
LAMBDA = 0.15;
convergence_threshold = 1;
num_points = length(DATA);
total_indices = [1:num_points];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DP Means - Initializations for algorithm %%%
% cluster count
K = 1;

% sets of points that make up clusters
L = {};
L = [L [1:num_points]];

% Class indicators/labels
Z = ones(1,num_points);

% means
MU = [];
MU = [MU; mean(DATA,1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializations for algorithm:
converged = 0;
t = 0;
while (converged == 0)
    t = t + 1;
    fprintf('Current iteration: %d...\n',t)
    
    %% Per Data Point:
    for i = 1:num_points
        
        %% CODE 1 - Calculate distance from current point to all currently existing clusters
        % Write code below here:
        
        %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
        % Write code below here:

    end
    
    %% CODE 3 - Form new sets of points (clusters)
    % Write code below here:
    
    %% CODE 4 - Recompute means per cluster
    % Write code below here:
    
    %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
    % Write code below here:
    
    %% CODE 6 - Plot final clusters after convergence 
    % Write code below here:
    
    if (converged)
        %%%%
    end    
end




