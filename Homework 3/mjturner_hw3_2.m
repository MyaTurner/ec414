% EC 414 - HW 3 - Spring 2022
% K-Means starter code

clear, clc, close all;

%% %%%%% 3.2 Generate Gaussian data and Implement k-means: %%%%% %%
%% 3.2a
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

% Create Scatter Plot
% First Gaussian Data
% figure
scatter(gaussian1(:,1), gaussian1(:, 2), 'r');
hold on
scatter(gaussian2(:,1), gaussian2(:, 2), 'g');
hold on
scatter(gaussian3(:,1), gaussian3(:, 2), 'b');
hold on

% title
title('My Gaussian Data')

% Initializations
k = 3;
converged = 0;
iteration = 0;
convergence_threshold = 0.025;
DATA = [gaussian1; gaussian2;  gaussian3];
% MU_init = [3 3; -4, -1; 2 -4];

% Creating iterator
% current_MU = MU_init;

%% 3.2b
% MU_init2 = [-0.14 2.6; 3.15 -0.84; -3.28 -1.58];

% Creating iterator
% current_MU = MU_init2;

%% 3.2c

% Initializations
wcss = [];
trailMUs = [];
iteration = 0;

for i = 1 : 10
    
    % Initialize 3 random centers
    MU_init = getRandom_centers(DATA, 3);
    num_mu = length(MU_init);
    
    % Creating iterator
    current_MU = MU_init;
    converged = 0;
    
    iteration = iteration + 1;
     fprintf('Iteration: %d\n',iteration)
    
    while (converged==0)
     
     %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
     labels = assignDatapoints(DATA, current_MU);
     
     %% CODE - Mean Updating - Update the cluster means
     newMU_init = recalculateCentriod(DATA, labels, num_mu);
     
     %% CODE 4 - Check for convergence 
     convergenceMetric = abs( sum( sum (newMU_init - current_MU) ) );
     if (convergenceMetric <= convergence_threshold)
         
         % Set converged to true
         converged=1;
         
         % If converged, get WCSS metric
         cost = WCSS(DATA, labels, current_MU, num_mu);
          
     else
         % If not converged, update current MU
          current_MU = newMU_init;
     end
    end
    wcss = [wcss cost];
    trailMUs = [trailMUs ; current_MU];
    
end

% %% K-Means implementation
% 
%  while (converged==0)
%       iteration = iteration + 1;
%       fprintf('Iteration: %d\n',iteration)
%       
%  
%      %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
%      labels = assignDatapoints(DATA, current_MU);
%      
%      %% CODE - Mean Updating - Update the cluster means
%      newMU_init = recalculateCentriod(DATA, labels, num_mu);
%      
%      %% CODE 4 - Check for convergence 
%      convergenceMetric = abs((newMU_init - current_MU) / current_MU);
%      if (convergenceMetric <= convergence_threshold)
%          converged=1;
%      end
%      
%      % If not converged, update current MU
%      current_MU = newMU_init;
%      
%      %% CODE 5 - Plot clustering results if converged:
%      
%      if (converged == 1)
%          fprintf('\nConverged.\n')
%          
%          % Getting points for each label
%          labeledData_1 = DATA((find(labels == 1)), :);
%          labeledData_2 = DATA((find(labels == 2)), :);
%          labeledData_3 = DATA((find(labels == 3)), :);
%          
%          % Create Scatter Plot
%          % figure
%          scatter(labeledData_1(:,1), labeledData_1(:, 2), 'r');
%          hold on
%          scatter(labeledData_2(:,1), labeledData_2(:, 2), 'g');
%          hold on
%          scatter(labeledData_3(:,1), labeledData_3(:, 2), 'b');
% 
%         % label axis and title
%         xlabel('First Feature')
%         ylabel('Second Feature')
%         title('k-means = 3 clusters')
%          
%          %% If converged, get WCSS metric
%          % Add code below
%          
%      end
%      
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  end  
%% Generate NBA data:
% % Add code below:
% 
% % HINT: readmatrix might be useful here
% 
%% Problem 3.2(f): Generate Concentric Rings Dataset using
% % sample_circle.m provided to you in the HW 3 folder on Blackboard.

