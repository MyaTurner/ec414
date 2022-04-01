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
lambda = 0.1;



theta = SGD(X_data_train, Y_label_train, tmax, lambda);


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
        
        tempY = Y_label_train;
        
        % Find places where y labels equal k and replace with 1
        yjEqualsk = find(Y_label_train == k);
        tempY(yjEqualsk) = 1;
        
        % Everywhere else place a 0
        yjNotEqualsk = find(Y_label_train ~= k);
        tempY(yjNotEqualsk) = 0;
        
        % Continue computing gradients
        probability = probablityNumerator / probabiltyDenominator;
        
        % Check on probability
        if probability < 10^(-10)
            probability = 10^(-10);
        end
        
        yjEqualsk = find(Y_label_train ~= k);
        v(:, k) = 2 * lambda + n * (probability - tempY) * xExt(:, j);
        
    end
    
    % Update parameters
    for k = 1 : m
        theta(:, k) = theta(:, k) - s_t * v(:, k);
    end
end

end
