% EC 414 - HW 3 - Spring 2022
% K-Means starter code

clear, clc, close all;

%% Generate Gaussian data:
% Add code below:

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

% %% Generate NBA data:
% % Add code below:
% 
% % HINT: readmatrix might be useful here
% 
% % Problem 3.2(f): Generate Concentric Rings Dataset using
% % sample_circle.m provided to you in the HW 3 folder on Blackboard.
% 
% %% K-Means implementation
% % Add code below
% 
% K =
% MU_init = 
% 
% MU_previous = MU_init;
% MU_current = MU_init;
% 
% % initializations
% labels = ones(length(DATA),1);
% converged = 0;
% iteration = 0;
% convergence_threshold = 0.025;
% 
% while (converged==0)
%     iteration = iteration + 1;
%     fprintf('Iteration: %d\n',iteration)
% 
%     %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
%     % Write code below here:
%     
%     %% CODE - Mean Updating - Update the cluster means
%     % Write code below here:
%     
%     %% CODE 4 - Check for convergence 
%     % Write code below here:
%     
%     if (something_here < convergence_threshold)
%         converged=1;
%     end
%     
%     %% CODE 5 - Plot clustering results if converged:
%     % Write code below here:
%     if (converged == 1)
%         fprintf('\nConverged.\n')
%         
%         
%        
%         
%         %% If converged, get WCSS metric
%         % Add code below
%         
%     end
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% end
% 
% 
% 
