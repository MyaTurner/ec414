% EC 414 - HW 3 - Spring 2022
% K-Means starter code

clear, clc, close all;

%% %%%%% 3.2d Heuristic choice of k via "elbow" method %%%%% %%
% Variables from 3.2a
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

% Initializing k range
k_array = [2 3 4 5 6 7 8 9 10];
DATA = [gaussian1; gaussian2;  gaussian3];
iteration = 0;
convergence_threshold = 0.025;
iteration = 0;
trials = 10;
bestWCSS_byk = [];

for k = 2 : 10
    fprintf('K value: %d\n',k)
    % Reset WCSS and formerly used indicies for each k
    alreadyUsed = [];
    wcss = [];
    for i = 1 : trials
        % Initialize 3 random centers
        [MU_init, usedIndices] = getRandom_centers(DATA, k, alreadyUsed);
    
        % Creating iterator
        current_MU = MU_init;
    
        % Resetting converged metric after every iteration
        converged = 0;
    
        % Making sure to use different random points at every iteration
        alreadyUsed = [alreadyUsed usedIndices];
        
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)
        innerIteration = 0;
        
        while(converged == 0)
            innerIteration = innerIteration + 1;
            fprintf('Iteration unitl next convergence: %d\n',innerIteration);
      
            %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
            labels = assignDatapoints(DATA, current_MU);
      
            %% CODE - Mean Updating - Update the cluster means
            newMU_init = recalculateCentriod(DATA, labels, k);
      
            %% CODE 4 - Check for convergence 
            convergenceMetric = abs( sum( sum (current_MU - newMU_init) ) );
            
            if(convergenceMetric <= convergence_threshold)
                
                % Set converged to true
                converged=1;
                
                fprintf('\nConverged.\n')
                
                % If converged, get WCSS metric
                cost = WCSS(DATA, labels, current_MU, k);
            else
                % If not converged, update current MU
                current_MU = newMU_init;
            end
        end
        wcss = [wcss cost];
    end
     % Find smallest WCSS
     minWCSS = min(wcss);
     
     % Add the best WCSS value for k to an array
     bestWCSS_byk = [bestWCSS_byk minWCSS];
end

% Plot results of WCSS v k range
plot(k_array, bestWCSS_byk, 'g');
xlabel('k range')
ylabel('WCSS values')
title('WCSS values v k range')


%% Problem 3.2(f): Generate Concentric Rings Dataset using
% % sample_circle.m provided to you in the HW 3 folder on Blackboard.

