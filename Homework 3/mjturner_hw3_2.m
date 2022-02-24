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

%% 3.2c Find best centers to intialize

% Initializations
wcss = [];
trialMUs = [];
alreadyUsed = [];
iteration = 0;
trials = 10;
num_mu = 3;

for i = 1 : trials
    
    % Initialize 3 random centers
    [MU_init, usedIndices] = getRandom_centers(DATA, num_mu, alreadyUsed);
    
    % Creating iterator
    current_MU = MU_init;
    
    % Resetting converged metric after every iteration
    converged = 0;
    
    % Making sure to use different random points at every iteration
    alreadyUsed = [alreadyUsed usedIndices];
    
    iteration = iteration + 1;
     fprintf('Iteration: %d\n',iteration)
     innerIteration = 0;
    
     while (converged == 0)
         innerIteration = innerIteration + 1;
         fprintf('Iteration unitl next convergence: %d\n',innerIteration);
      
      %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
      labels = assignDatapoints(DATA, current_MU);
      
      %% CODE - Mean Updating - Update the cluster means
      newMU_init = recalculateCentriod(DATA, labels, num_mu);
      
      %% CODE 4 - Check for convergence 
      convergenceMetric = abs( sum( sum (current_MU - newMU_init) ) );
      if (convergenceMetric <= convergence_threshold)
          
          % Set converged to true
          converged=1;
          
          % If converged, get WCSS metric
          cost = WCSS(DATA, labels, current_MU, num_mu);
          
          fprintf('\nConverged.\n')
          
          % Getting points for each label
         labeledData_1 = DATA((find(labels == 1)), :);
          labeledData_2 = DATA((find(labels == 2)), :);
          labeledData_3 = DATA((find(labels == 3)), :);
          
          % Create Scatter Plot
           figure
          scatter(labeledData_1(:,1), labeledData_1(:, 2), 'r');
          hold on
          scatter(labeledData_2(:,1), labeledData_2(:, 2), 'g');
          hold on
          scatter(labeledData_3(:,1), labeledData_3(:, 2), 'b');
 
         % label axis and title
         xlabel('First Feature')
         ylabel('Second Feature')
         title (sprintf('Trial %d',i))
           
      else
          % If not converged, update current MU
           current_MU = newMU_init;
      end
     end
     wcss = [wcss cost];
     trialMUs = [trialMUs ; MU_init];
  
end

% Find smallest WCSS
minWCSS = min(wcss);

% Finding clusters the smallest WCSS produced
bestTrial = find(wcss == minWCSS, 1);
fprintf('Trial %d had the best wcss with a value of %d\n',bestTrial, floor(minWCSS) );
index = 3 * bestTrial;
bestIntial_MUs = trialMUs(index : index + 2, :);

% Creating iterator
 current_MU = bestIntial_MUs;

% Plotting the WCSS

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

