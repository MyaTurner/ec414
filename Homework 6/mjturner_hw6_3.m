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


%% Functions

function [outputArg1,outputArg2] = uh(inputArg1,inputArg2)
%UH Summary of this function goes here
%   Detailed explanation goes here
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end
