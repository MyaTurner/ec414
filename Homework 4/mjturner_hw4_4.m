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
offset_array = zeros(1, 9);
scaling_array = zeros(1,9);

% Title for code output of means and stand deviations of training set
fprintf('The mean and standard deviations of training data before normalization\n\n')

% Finding means and standara deviation before normalization
for i = 1: length(names)

    offset_array(i) = mean(ytrain);
    if( i == 9)
        fprintf('The mean of label column %s is: %.3f \n',char(names(i)), mean(ytrain));
        fprintf('The standard deviation of label column %s is: %.3f \n\n',char(names(i)), std(ytrain));
        offset_array(i) = mean(ytrain);
        scaling_array(i) = std(ytrain);
        break
    end

    feature = Xtrain(:, i);
    fprintf('The mean of feature column %s is: %.3f \n',char(names(i)), mean(feature));
    fprintf('The standard deviation of feature column %s is: %.3f \n\n',char(names(i)), std(feature));
    offset_array(i) = mean(feature);
    scaling_array(i) = std(feature);
end

% Normalize training data
Xtrain = normalize(Xtrain);

% 4.4(b)
e_vals = [-5 : 10];
