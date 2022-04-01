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

for t = 20: 20: tmax
        
        % Find theta
        currentTheta = SGD(X_data_train, Y_label_train, t, lambda);
        currentG = (1 / n) * regularizedLogisticloss(X_data_train, Y_label_train, currentTheta, lambda);
        gTheta = [currentG gTheta];

end

figure(3)
plot(20 : 20 : 6000, gTheta);
title('Regularized Logistic Loss Normalized by Total of Training Examples')
xlabel('20t')
ylabel('Regularized Logistic Loss Normalized')



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
        probablityNumerator = exp(theta(:, k)') * xExt(:, j) ;
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
   
        v(:, k) = 2 * lambda + n * (probability - Y_label_train(j)) .* xExt(:, j);
        
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
        metric = Y_label_train(l) * theta(:, l)' * xExt(:, j);
        secondSum = secondSum + metric;
    end
    
    subtrahend = secondSum;

    fj(j) = minuend - subtrahend;
end

% Find g(theta)
gTheta = f0 + sum(fj);

end

