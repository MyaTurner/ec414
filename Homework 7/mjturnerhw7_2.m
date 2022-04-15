%% Homework 7.2

% Load iris dataset
load("iris.mat")

%% Pre-work before 7.2a
Xtrain = [X_data_train(:, 2) X_data_train(:, 4)];
Xtest = [X_data_test(:, 2) X_data_test(:, 4);];
tmax = 2 * 10^5;
C = 1.2;

% Creating Class variables (Training)
% Class 1 v Class 2
Data1v2 = Xtrain(find(Y_label_train == 1 | Y_label_train == 2 ), :);
Data1v2_labels = Y_label_train(Y_label_train == 1 | Y_label_train == 2); % Class 1 will be +1
Data1v2_labels(Y_label_train == 2) = -1; % Class 2 will be -1

% Class 2 v Class 3
Data2v3 = Xtrain(find(Y_label_train == 2 | Y_label_train == 3 ), :);
Data2v3_labels = Y_label_train(Y_label_train == 2 | Y_label_train == 3);
Data2v3_labels(Y_label_train == 2) = +1; % Class 2 will be +1
Data2v3_labels(Y_label_train == 3) = -1; % Class 3 will be -1


% Class 1 v Class 3
Data1v3 = Xtrain(find(Y_label_train == 1 | Y_label_train == 3 ), :);
Data1v3_labels = Y_label_train(Y_label_train == 1 | Y_label_train == 3); % Class 1 will be +1
Data1v3_labels(Y_label_train == 3) = -1; % Class 3 will be -1

% Creating Classs variables (Test)
% Class 1 v Class 2
Data1v2_test = Xtest(find(Y_label_test == 1 | Y_label_test == 2 ), :);
Data1v2_labels_test = Y_label_test(Y_label_test == 1 | Y_label_test == 2); % Class 1 will be +1
Data1v2_labels_test(Y_label_test == 2) = -1; % Class 2 will be -1

% Class 2 v Class 3
Data2v3_test = Xtest(find(Y_label_test == 2 | Y_label_test == 3 ), :);
Data2v3_labels_test = Y_label_test(Y_label_test == 2 | Y_label_test == 3);
Data2v3_labels_test(Y_label_test == 2) = +1; % Class 2 will be +1
Data2v3_labels_test(Y_label_test == 3) = -1; % Class 3 will be -1


% Class 1 v Class 3
Data1v3_test = Xtest(find(Y_label_test == 1 | Y_label_test == 3 ), :);
Data1v3_labels_test = Y_label_test(Y_label_test == 1 | Y_label_test == 3); % Class 1 will be +1
Data1v3_labels_test(Y_label_test == 3) = -1; % Class 3 will be -1




% Finding theta or w for SVM Binary classifiers
theta_array = [];
classes = [Data1v2 Data2v3 Data1v3];
classesLabels = [Data1v2_labels Data2v3_labels Data1v3_labels];

for i = 1 : 3
    theta = SSGD(classes(i), classesLabels(i), tmax, C);
    theta_array = [theta_array theta];
end

%% 7.2a
% Find sample-normalized cost for each binary classifier
cost_array = [];

for i = 1 : 3
    
    for t = 1 : 1000 : 200000
        [n, ~] = size(classes(i));
        cost = (1 / n) * cost(classes(i), classesLabels(i), theta_array(i), C);
        cost_array = [cost_array cost];
    end
    
    % Plot sample-normalized cost for each binary classifier
    figure(i)
    plot(1 : 1000 : 200000, cost_array);
    plotName = sprintf('Sample Normalized Cost for Binary Classifier %d', i);
    ylabelName = sprintf('Binary Classifier %d', i);
    title(plotName)
    xlabel('t')
    ylabel('Sample Normalized Cost')
end

%% 7.2b

% Find training CCR

 



%% Functions (NEED EDITS FOR 7.2)

function theta = SSGD(Xtrain, Y_label_train, tmax, C)

% Stochastic Sub-Gradient Descent for Soft Margin Binary SVM

% Initializing variables
[n, d] = size(Xtrain); % Rd space is for the feature vectors and n is the number of examples
% Note m is the number of unique labels present
m = height(unique(Y_label_train));
theta = zeros(d + 1, 1);
% Note a extended vector has an extra one at d + 1 position
xExt = [Xtrain; ones(1, d)];
% Creating a extended identity matrix
id = [ones(1, d) 0];
idExt = diag(id);

for t = 1 : tmax
    
    % Choose sample index
    j = randsample(range, 1);
    
    % Initialize st
    s_t = 0.5 / t;
    
    % Compute gradients
    v = idExt * theta;
    
    if( Y_label_train(j) * theta' * xExt(:, j) < 1 )
        v = v - n * C * Y_label_train(j) * xExt(:, j);
    end
    
    % Update parameters
    theta = theta - s_t * v;

    
end

end

function gTheta = cost(X_data_train, Y_label_train, theta, C)

% Find the cost function
% Initializing variables
[n, d] = size(X_data_train);
% Note a extended vector has an extra one at d + 1 position
xExt = [X_data_train; ones(1, d)];

% Find f0(theta) % Do not know what is w, assuming is theta
f0 = (1 / 2) * norm(theta).^2;

% Find fj(theta)
fj = zeros(1, d);
for j = 1 : n
    fj = C * max([0 (1 - Y_label_train(j) * theta' * xExt(:, j) )]);
end

% Find g(theta)
gTheta = f0 + sum(fj);

end

function [ccr, yj] = CCR(X_data_train, Y_label_train, theta)

% Find CCR of the training set
% Initializing variables
[n, d] = size(X_data_train);
% Note a extended vector has an extra one at d + 1 position
xExt = [X_data_train; ones(1, d)];

% Find the decision rule for a binary SVM
yj = zeros(1, d);
for j = 1 : d
    yj(j) = sign( theta' * xExt(:, j) );
end
    
% CCR
ccr = (1 / n) * sum(Y_label_train == yj);
end