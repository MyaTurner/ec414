% EC 414 - HW 3 - Spring 2022
% K-Means starter code

clear, clc, close all;

%% %%%%% 3.2f Failure of K Means %%%%% %%

% Getting sample circle data
DATA = sample_circle(3);

% Initial plot of sample circle data
figure
scatter(DATA(1:500, 1), DATA(1:500, 2))
hold on
scatter(DATA(501:1000, 1), DATA(501:1000, 2))
hold on
scatter(DATA(1001:1500, 1), DATA(1001:1500, 2))
xlabel('First Feature')
ylabel('Second Feature')
title('Sample Circle Data Plot')

% Initializing k range
k = 3;
iteration = 0;
convergence_threshold = 0.025;
iteration = 0;
trials = 10;
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
    
     while (converged == 0)
         innerIteration = innerIteration + 1;
         fprintf('Iteration unitl next convergence: %d\n',innerIteration);
      
      %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
      labels = assignDatapoints(DATA, current_MU);
      
      %% CODE - Mean Updating - Update the cluster means
      newMU_init = recalculateCentriod(DATA, labels, k);
      
      %% CODE 4 - Check for convergence 
      convergenceMetric = abs( sum( sum (current_MU - newMU_init) ) );
      if (convergenceMetric <= convergence_threshold)
          
          % Set converged to true
          converged=1;
          
          % If converged, get WCSS metric
          cost = WCSS(DATA, labels, current_MU, k);
          
          fprintf('\nConverged.\n')
          
          % Getting points for each label
          labeledData_1 = DATA((find(labels == 1)), :);
          labeledData_2 = DATA((find(labels == 2)), :);
          labeledData_3 = DATA((find(labels == 3)), :);
          
          
          % Create Scatter Plot
          figure
          scatter(labeledData_1(:,1), labeledData_1(:, 2));
          hold on
          scatter(labeledData_2(:,1), labeledData_2(:, 2));
          hold on
          scatter(labeledData_3(:,1), labeledData_3(:, 2));
        
 
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
  
end

% Find smallest WCSS
minWCSS = min(wcss);

% Finding clusters the smallest WCSS produced
bestTrial = find(wcss == minWCSS, 1);
fprintf('Trial %d had the best wcss with a value of %d\n',bestTrial, floor(minWCSS) );

