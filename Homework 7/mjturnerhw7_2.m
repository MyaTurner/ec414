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
Data1v2_labels(Data1v2_labels == 2) = -1; % Class 2 will be -1

% Class 2 v Class 3
Data2v3 = Xtrain(find(Y_label_train == 2 | Y_label_train == 3 ), :);
Data2v3_labels = Y_label_train(Y_label_train == 2 | Y_label_train == 3);
Data2v3_labels(Data2v3_labels == 2) = +1; % Class 2 will be +1
Data2v3_labels(Data2v3_labels == 3) = -1; % Class 3 will be -1

% Class 1 v Class 3
Data1v3 = Xtrain(find(Y_label_train == 1 | Y_label_train == 3 ), :);
Data1v3_labels = Y_label_train(Y_label_train == 1 | Y_label_train == 3); % Class 1 will be +1
Data1v3_labels(Data1v3_labels == 3) = -1; % Class 3 will be -1

% Creating Classs variables (Test)
% Class 1 v Class 2
Data1v2_test = Xtest(find(Y_label_test == 1 | Y_label_test == 2 ), :);
Data1v2_labels_test = Y_label_test(Y_label_test == 1 | Y_label_test == 2); % Class 1 will be +1
Data1v2_labels_test(Data1v2_labels_test == 2) = -1; % Class 2 will be -1

% Class 2 v Class 3
Data2v3_test = Xtest(find(Y_label_test == 2 | Y_label_test == 3 ), :);
Data2v3_labels_test = Y_label_test(Y_label_test == 2 | Y_label_test == 3);
Data2v3_labels_test(Data2v3_labels_test == 2) = +1; % Class 2 will be +1
Data2v3_labels_test(Data2v3_labels_test == 3) = -1; % Class 3 will be -1


% Class 1 v Class 3
Data1v3_test = Xtest(find(Y_label_test == 1 | Y_label_test == 3 ), :);
Data1v3_labels_test = Y_label_test(Y_label_test == 1 | Y_label_test == 3); % Class 1 will be +1
Data1v3_labels_test(Data1v3_labels_test == 3) = -1; % Class 3 will be -1


% Finding theta or w for SVM Binary classifiers (Training)
theta_array = [];
classes_train = [Data1v2 Data2v3 Data1v3];
classesLabels_train = [Data1v2_labels Data2v3_labels Data1v3_labels];

 j = 1;
for i = 1 : 2 : 6

    theta = SSGD(classes_train(:, i : i + 1), classesLabels_train(:, j), tmax, C);
    theta_array = [theta_array theta];
    j = j + 1;
end

%% 7.2a
% Find sample-normalized cost for each binary classifier
cost_array = [];
j = 1;
for i = 1 : 2 : 6
    
    for t = 1 : 1000 : 200000
        
        [n, ~] = size(classes_train(:, i : i + 1));
        cost = (1 / n) * costFunction(classes_train(:, i : i + 1), classesLabels_train(:, j), theta_array(:, j), C);
        cost_array = [cost_array cost];
    end
   
    % Plot sample-normalized cost for each binary classifier
    figure(j)
    plot(1 : 1000 : 200000, cost_array);
    plotName = sprintf('Sample Normalized Cost for Binary Classifier %d', j);
    title(plotName)
    xlabel('t')
    ylabel('Sample Normalized Cost')
    cost_array = [];
     j = j + 1;
end

%% 7.2b
% Find training CCR
ccr_training = [];
ccr_array_training = [];
predictedLabels_array_training = [];
j = 1;
 
for i = 1 : 2 : 6
    
    for t = 1 : 1000 : 200000
        [n, ~] = size(classes_train(:, i : i + 1));
        [ccr, predictedLabels_training] = CCR(classes_train(:, i : i + 1), classesLabels_train(:, j), theta_array(:, j));
        ccr_training = [ccr_training ccr];
    end
   
    
    % Plot the training ccr for each binary classifier
    figure(3 + j)
    plot(1 : 1000 : 200000, ccr_training);
    plotName = sprintf('CCR Training for Binary Classifier %d', j);
    title(plotName)
    xlabel('t')
    ylabel('CCR Training')
    ccr_array_training = [ccr_array_training ccr_training];
    ccr_training = [];
    predictedLabels_array_training = [predictedLabels_array_training  predictedLabels_training];
     j = j + 1;
end

%% 7.2c
classes_test = [Data1v2_test Data2v3_test Data1v3_test];
classesLabels_test = [Data1v2_labels_test Data2v3_labels_test Data1v3_labels_test];


% Find test CCR
ccr_test = [];
ccr_array_test = [];
predictedLabels_array_test = [];
 j = 1;
 
for i = 1 : 2 : 6
    
    for t = 1 : 1000 : 200000
        [n, ~] = size(classes_test(:, i : i + 1));
        [ccr, predictedLabels_test] = CCR(classes_test(:, i : i + 1), classesLabels_test(:, j), theta_array(:, j));
        ccr_test = [ccr_test ccr];
    end
 
    
    % Plot the test ccr for each binary classifier
    figure(6 + j)
    plot(1 : 1000 : 200000, ccr_test);
    plotName = sprintf('CCR Test for Binary Classifier %d', j);
    title(plotName)
    xlabel('t')
    ylabel('CCR Test')
    ccr_array_test = [ccr_array_test ccr_test];
    ccr_test = [];
    predictedLabels_array_test = [predictedLabels_array_test predictedLabels_test];
        j = j + 1;

end

%% 7.2d
% Reporting final values
% theta
for i = 1 : 3
    fprintf('Final Theta for Binary Classifier %d  \n', i)
    disp(theta_array(:, i))
end

% The training CCR
for i = 1 : 3
    fprintf('\n Final training CCR for Binary Classifier %d  \n', i)
    disp(ccr_array_training(end, i))
end

% the test CCR
for i = 1 : 3
    fprintf('\n Final test CCR for Binary Classifier %d \n', i)
    disp(ccr_array_test(end, i))
end

% Trainig confusion matrix (work on)
for i = 1 : 3
    fprintf('\n Training Confusion matrix for Binary Classifier %d \n', i)
    confusionmatTraining = confusionmat(classesLabels_train(:, i), predictedLabels_array_training(:, i));
    disp(confusionmatTraining)
end

% Test confusion matrix (work on)
for i = 1 : 3
    fprintf('\n Test Confusion matrix for Binary Classifier %d \n', i)
    confusionmatTest = confusionmat(classesLabels_test(:, i), predictedLabels_array_test(:, i));
    disp(confusionmatTest)
end

%% 7.2e
% Implement All Pairs Methodology (Training)
[n, d] = size(classesLabels_train);
yAllPairs_training = zeros(n, d);

for j = 1 : d
    
    for i = 1 : n
        classfier = yAllPairs_training(:, j);
        
        % Find the classifer for Data1v2.  Note Class 1 is + 1 and Class 2 is -1.
        if(j == 1)
            % Should give back 1 when ytrain is 1 and then 2 when ytrain is -1.
            classfier(i) = -(1 / 2) * classesLabels_train(i, j) + (3 / 2);
        end
        
        % Find the classifer for Data2v3.  Note Class 2 is + 1 and Class 3 is -1.
        if(j == 2)
            % Should give back 1 when ytrain is 2 and then 3 when ytrain is -1.
            classfier(i) = -(1 / 2) * classesLabels_train(i, j) + (5 / 2);
        end
        
        % Find the classifer for Data1v3.  Note Class 1 is + 1 and Class 3 is -1.
        if(j == 3)
            % Should give back 1 when ytrain is 1 and then 3 when ytrain is -1.
            classfier(i) = -1 * classesLabels_train(i, j) + 2;
        end
        
    end
    
end

yAllPairs_training = mode(yAllPairs_training')';

% Implement All Pairs Methodology (Test)
[n, d] = size(classesLabels_test);
yAllPairs_test = zeros(n, d);

for j = 1 : d
    
    for i = 1 : n
        classfier = yAllPairs_test(:, j);
        
        % Find the classifer for Data1v2.  Note Class 1 is + 1 and Class 2 is -1.
        if(j == 1)
            % Should give back 1 when ytrain is 1 and then 2 when ytrain is -1.
            classfier(i) = -(1 / 2) * classesLabels_test(i, j) + (3 / 2);
        end
        
        % Find the classifer for Data2v3.  Note Class 2 is + 1 and Class 3 is -1.
        if(j == 2)
            % Should give back 1 when ytrain is 2 and then 3 when ytrain is -1.
            classfier(i) = -(1 / 2) * classesLabels_test(i, j) + (5 / 2);
        end
        
        % Find the classifer for Data1v3.  Note Class 1 is + 1 and Class 3 is -1.
        if(j == 3)
            % Should give back 1 when ytrain is 1 and then 3 when ytrain is -1.
            classfier(i) = -1 * classesLabels_test(i, j) + 2;
        end
        
    end
    
end

yAllPairs_test = mode(yAllPairs_test')';


% Find training CCR
ccr_training = [];
ccr_array_training = [];
predictedLabels_array_training = [];
j = 1;
 
for i = 1 : 2 : 6
    
    for t = 1 : 1000 : 200000
        [n, ~] = size(classes_train(:, i : i + 1));
        [ccr, predictedLabels_training] = CCR(classes_train(:, i : i + 1), yAllPairs_training, theta_array(:, j));
        ccr_training = [ccr_training ccr];
    end
    j = j + 1;
    
    ccr_array_training = [ccr_array_training ccr_training];
    ccr_training = [];
    predictedLabels_array_training = [predictedLabels_array_training  predictedLabels_training];
end

% Find the test ccr
j = 1;
for i = 1 : 2 : 6
    
    for t = 1 : 1000 : 200000
        [n, ~] = size(classes_test(:, i : i + 1));
        [ccr, predictedLabels_test] = CCR(classes_test(:, i : i + 1), yAllPairs_test, theta_array(:, j));
        ccr_test = [ccr_test ccr];
    end
     j = j + 1;
    
    ccr_array_test = [ccr_array_test ccr_test];
    ccr_test = [];
    predictedLabels_array_test = [predictedLabels_array_test predictedLabels_test];

end

% Reporting final values
% The training CCR
for i = 1 : 3
    fprintf('\n Final training CCR (All Pairs Method) for Binary Classifier %d  \n', i)
    disp(ccr_array_training(end, i))
end

% the test CCR
for i = 1 : 3
    fprintf('\n Final test CCR (All Pairs Method) for Binary Classifier %d \n', i)
    disp(ccr_array_test(end, i))
end

% Trainig confusion matrix (work on)
for i = 1 : 3
    fprintf('\n Training Confusion matrix (All Pairs Method) for Binary Classifier %d \n', i)
    confusionmatTraining = confusionmat(yAllPairs_training, predictedLabels_array_training(:, i));
    disp(confusionmatTraining)
end

% Test confusion matrix (work on)
for i = 1 : 3
    fprintf('\n Test Confusion matrix (All Pairs Method) for Binary Classifier %d \n', i)
    confusionmatTest = confusionmat(yAllPairs_test, predictedLabels_array_test(:, i));
    disp(confusionmatTest)
end


%% Functions

function theta = SSGD(Xtrain, Y_label_train, tmax, C)

% Stochastic Sub-Gradient Descent for Soft Margin Binary SVM

% Initializing variables
[n, d] = size(Xtrain); % Rd space is for the feature vectors and n is the number of examples
% Note m is the number of unique labels present
m = height(unique(Y_label_train));
theta = zeros(d + 1, 1);
range = 1 : n;
% Note a extended vector has an extra one at n + 1 position
xExt = [Xtrain ones(n, 1)];
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
    
    if( Y_label_train(j) * theta' * xExt(j, :)' < 1 )
        v = v - n * C * Y_label_train(j) * xExt(j, :)';
    end
    
    % Update parameters
    theta = theta - s_t * v;

    
end

end

function gTheta = costFunction(X_data_train, Y_label_train, theta, C)

% Find the cost function
% Initializing variables
[n, d] = size(X_data_train);
% Note a extended vector has an extra one at n + 1 position
xExt = [X_data_train ones(n, 1)];

% Find f0(theta) % Do not know what is w, assuming is theta
f0 = (1 / 2) * norm(theta).^2;

% Find fj(theta)
fj = zeros(1, d);
for j = 1 : n
    
    if( 1 - Y_label_train(j) * theta' * xExt(j, :)' < 0 )
        MAX = 0;
    else
        MAX = 1 - Y_label_train(j) * theta' * xExt(j, :)';
    end
    
    fj = C * MAX;
end

% Find g(theta)
gTheta = f0 + sum(fj);

end

function [ccr, y_hat] = CCR(X_data_train, Y_label_train, theta)

% Find CCR of the training set
% Initializing variables
[n, d] = size(X_data_train);
% Note a extended vector has an extra one at d + 1 position
xExt = [X_data_train ones(n, 1)];

% Find the decision rule for a binary SVM
y_hat = zeros(n, 1);
for j = 1 : n
    y_hat(j) = sign( theta' * xExt(j, :)' );
end
    
% CCR
ccr = (1 / n) * sum(Y_label_train == y_hat);
end