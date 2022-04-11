%% Homework 7.2

% Load iris dataset
load("iris.mat")

% Need variables to test the whole dataset
yLabels = [Y_label_test; Y_label_train];
xData = [X_data_test; X_data_train];
[d n] = size(xData);

%% Functions (NEED EDITS FOR 7.2)

function theta = SGD(X_data_train, Y_label_train, tmax, lambda)

% Stochastic Gradient Descent for regularized multi-class Logistic-Loss

% Initializing variables
[d n] = size(X_data_train);
% Note m is the number of unique labels present
m = height(unique(Y_label_train));
theta = zeros(d + 1, m);
range = 1 : n;
% Note a extended vector has an extra one at d + 1 position
xExt = [X_data_train; ones(1, n)];

for t = 1 : tmax
    % Choose sample index
    j = randsample(range, 1);
    
    % Initialize st
    s_t = 0.01 / t;
    
    % Compute gradients
    for k = 1 : m
        probablityNumerator = exp( theta(:, k)') * xExt(:, j);
        probabiltyDenominator = 0;
        for l = 1 : m
            temp = exp(theta(:, l)') * xExt(:, j);
            probabiltyDenominator = temp + probabiltyDenominator;
        end
        
        % Continue computing gradients
        probability = probablityNumerator / probabiltyDenominator;
        
        % Check on probability
        if probability < 10^(-10)
            probability = 10^(-10);
        end
   
        v(:, k) = 2 * lambda + n * (probability - (k == Y_label_train(j)) ) .* xExt(:, j);
        
    end
    
    % Update parameters
    for k = 1 : m
        theta(:, k) = theta(:, k) - s_t * v(:, k);
    end
end

end

function gTheta = regularizedLogisticloss(X_data_train, Y_label_train, theta, lambda)

% Find the regularized logistic loss
% Initializing variables
[~, n] = size(X_data_train);
% Note m is the number of unique labels present
m = height(unique(Y_label_train));
% Note a extended vector has an extra one at d + 1 position
xExt = [X_data_train; ones(1, n)];

summation = 0;
% Find f0(theta)
for l = 1 : m
    distanceSquared= norm(theta(:, l)) ^ 2;
    summation = summation + distanceSquared;
    
end

f0 = lambda * summation;

% Find fj(theta)
fj = zeros(1, n);
for j = 1 : n
    
    % Find minued in fj
    firstSum = 0;
    for l = 1 : m
        metric = exp( theta(:, l)' * xExt(:, j) );
        firstSum = firstSum + metric;
    end

    minuend = log(firstSum);

    % Find subtrahend in fj
    secondSum = 0;
    for l = 1 : m
        metric = ( l == Y_label_train(j) ) * theta(:, l)' * xExt(:, j);
        secondSum = secondSum + metric;
    end
    
    subtrahend = secondSum;

    fj(j) = minuend - subtrahend;
end

% Find g(theta)
gTheta = f0 + sum(fj);

end

function [ccr, yj] = CCR(X_data_train, Y_label_train, theta)

% Find CCR of the training set
% Initializing variables
[~, n] = size(X_data_train);
% Note a extended vector has an extra one at d + 1 position
xExt = [X_data_train; ones(1, n)];
% Note m is the number of unique labels present
m = height(unique(Y_label_train));
[r, ~] = size(Y_label_train);

% Find the decision rule in Logistic Regression
yj = zeros(r, 1);
for j = 1 : n
    decisionRule = zeros(1, m);
    for l = 1 : m
        decisionRule(l) = theta(:, l)' * xExt(:, j);
    end
    [~, finalRule] = max(decisionRule);
    yj(j) = finalRule;
end
    
% CCR
ccr = (1 / n) * sum(Y_label_train == yj);
end

function logLoss_theta = logLoss(X_data_test, Y_label_test, theta)

% Find the log-loss of a data set
%Initialize variables
[~, n] = size(X_data_test);
m = length(Y_label_test);
% Note a extended vector has an extra one at d + 1 position
xExt = [X_data_test; ones(1, n)];

% Find the probability for yj give xj of theta
summation = zeros(1, n);
for j = 1 : n
    
        probablityNumerator = exp( theta(:, Y_label_test(j))' .* xExt(:, j) ) ;
        probabiltyDenominator = 0;
        for l = 1 : 3
            temp = exp(theta(:, l)' .* xExt(:, j) );
            probabiltyDenominator = temp + probabiltyDenominator;
        end
        
        % Continue computing gradients
        probability = probablityNumerator / probabiltyDenominator;
        
        % Check on probability
        if probability < 10^(-10)
            probability = 10^(-10);
        end
    summation(j) = log(probability);
end

logLoss_theta = -(1 / n) * sum(summation);
end
