%% 6.3a

% Load iris dataset
load("iris.mat")

% Need variables to test the whole dataset
yLabels = [Y_label_test; Y_label_train];
xData = [X_data_test; X_data_train];
[d n] = size(xData);

% (i) Plot histogram of class labels (train + test dataset).
figure(1)
hold on
histogram(yLabels)
title("Class Labels")
xlabel("Label")
ylabel("Amount of Each Label ")
hold off

% (ii) Compute the matrix of empirical correlation coefficients for all pairs of features (train + test dataset).
for i = 1 : n
    for j = (i + 1) : n
        feature1 = xData(:, i);
        feature2 = xData(:, j);
        correlationCoefficient = corrcoef(feature1, feature2);
        fprintf('Empirical Correlation Coefficients between features %d and %d:\n', i, j)
        disp(correlationCoefficient)
    end
end

% Newline
fprintf("\n")

% (iii) Create 2D scatter plots of the dataset for all distinct pairs of features.
plotNumber = 1;
figure(2)
sgtitle('2D Scatter Plots of the Dataset for All Distinct Pairs of Features')
hold on
for i = 1 : n
    for j = (i + 1) : n
        
        feature1 = xData(:, i);
        feature2 = xData(:, j);
        subplot(3, 2, plotNumber)
        scatter(feature1, feature2)
        plotName = sprintf('Feature %d vs. Feature %d', j, i);
        xlabelName = sprintf('Feature %d', i);
        ylabelName = sprintf('Feature %d', j);
        title(plotName)
        xlabel(xlabelName)
        ylabel(ylabelName)
        plotNumber = plotNumber + 1;
    end
end

hold off

%% 6.3b
tmax = 6000;
tbatch = 20;
lambda = 0.1;
gTheta = [];
[d n] = size(X_data_train);
ccr_training = [];
ccr_test = [];
logLoss_theta = [];

for t = 20: 20: tmax
        
        % Find theta
        currentTheta = SGD(X_data_train, Y_label_train, t, lambda);
        currentG = (1 / n) * regularizedLogisticloss(X_data_train, Y_label_train, currentTheta, lambda);
        gTheta = [gTheta currentG];
        
        % Finding CCR of training data for 6.3c
        [currentCCRtrain, predictedLabels_training] = CCR(X_data_train, Y_label_train, currentTheta);
        ccr_training = [ccr_training currentCCRtrain];
        
        % Finding CCR of test data for 6.3d
        thetaTest = SGD(X_data_test, Y_label_test, t, lambda);
        [currentCCRtest, predictedLabels_test]  = CCR(X_data_test, Y_label_test, thetaTest);
        ccr_test = [ccr_test currentCCRtest];
        
%         % Finding CCR of test data for 6.3e
%         currentLoss = logLoss(X_data_test, Y_label_test, theta);
%         logLoss_theta = [logLoss_theta currentLoss];
        

end

figure(3)
plot(20 : 20 : tmax, gTheta);
title('Regularized Logistic Loss Normalized by Total of Training Examples')
xlabel('t')
ylabel('Regularized Logistic Loss Normalized')

%% 6.3c
figure(4)
plot(20 : 20 : tmax, ccr_training);
title('CCR of the training set v. Iteration')
xlabel('t')
ylabel('CCR of Training Data')

%% 6.3d
figure(5)
plot(20 : 20 : tmax, ccr_test);
title('CCR of the Test set v. Iteration')
xlabel('t')
ylabel('CCR of Test')

%% 6.3e
% figure(6)
% plot(20 : 20 : tmax, logLoss_theta);
% title('Log Loss of the Test Set v. Iteration')
% xlabel('t')
% ylabel('Log Loss')

%% 6.3f

% Reporting final values
% (i) theta
fprintf('Final Theta \n')
disp(theta)

% (ii) The training CCR
fprintf('\n Final training CCR \n')
disp(ccr_training(end))

% (iii) the test CCR
fprintf('\n Final test CCR \n')
disp(ccr_test(end))

% (iv) Trainig confusion matrix
fprintf('\n Training Confusion matrix \n')
confusionmatTraining = confusionmat(Y_label_train, predictedLabels_training);
disp(confusionmatTraining)

% (v) Test confusion matrix
fprintf('\n Test Confusion matrix \n')
confusionmatTest = confusionmat(Y_label_test, predictedLabels_test);
disp(confusionmatTraining)

%% Functions

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
