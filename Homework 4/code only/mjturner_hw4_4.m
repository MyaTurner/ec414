%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 414 (Ishwar) Spring 2022
% HW 4.4
% Mya Turner: mjturner@bu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 4.4(a)
clear all; clc;
rng('default')  % For reproducibility of data and results

data = load('prostateStnd.mat');

% Defining data
Xtrain = data.Xtrain;
ytrain = data.ytrain;
Xtest = data.Xtest;
ytest = data.ytest;
names = data.names;
num_features = 8;
offset_array = zeros(1, 9);
scaling_array = zeros(1,9);

% Title for code output of means and stand deviations of training set
fprintf('The mean and variance of training data before normalization\n\n')

% Finding means and standara deviation before normalization
for i = 1: length(names)

    offset_array(i) = mean(ytrain);
    if( i == 9)
        fprintf('The mean of label column %s is: %.3f \n',char(names(i)), mean(ytrain));
        fprintf('The variance of label column %s is: %.3f \n\n',char(names(i)), var(ytrain));
        offset_array(i) = mean(ytrain);
        scaling_array(i) = std(ytrain);
        break
    end

    feature = Xtrain(:, i);
    fprintf('The mean of feature column %s is: %.3f \n',char(names(i)), mean(feature));
    fprintf('The variance of feature column %s is: %.3f \n\n',char(names(i)), var(feature));
    offset_array(i) = mean(feature);
    scaling_array(i) = std(feature);
end

% Normalize training data
Xtrain = normalize(Xtrain);
ytrain = normalize(ytrain);

% 4.4(b)
e_vals = [-5 : 10];

for i = 1 : length(e_vals)
[w_ridge_array, b_ridge_array] = Ridge(Xtrain, ytrain, num_labels, e_vals(i));
end

%Ran out of time to test 4.4b, however wrote down my logic.

% 4.4(c)
% Again ran out of time to plot however this would be my code to do it.
figure()
hold on
grid;axis equal;hold on;
title('Ridge regresssion v lambda');
for i = 1 : length(e_vals)
    
    plot(w_ridge_array, [-5:10], 'linewidth', 2);
    hold on
end
legend('feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5', 'feature 6','feature 7', 'feature 8')
title('CCR v b')
hold off
axis equal;





% Find functions below!

function [w_ridge_array, b_ridge_array] = Ridge(Xtrain, ytrain, num_labels, e)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For each class compute w_ridge and b_ridge
muy = mean(ytrain);
Y_centered = ytrain - muy';
if(exp(e) > 0)
    for i = 1: num_labels

        % Get current feature
        X = Xtrain(:, i);
        n = length(X);

        % Step 1) Find empirical means for each class
        mux = mean(X);

        % Step 3) Find covariance matrices
        % Need to get centered means
        X_centered = X - mux';
        sigma_x  = 1/n * X_centered * X_centered';

        sigma_xy = 1/n * X_centered * Y_centered';

        
        w_ridge_array(i) = inv ((exp(e) / n * eye(67) + sigma_x)) * sigma_xy;
        b_ridge_array(i) = muy - (w_ridge_array(i))' * mux;
        

    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
