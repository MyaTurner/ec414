%% Homework 8.2

% Load dataset
load('kernel-svm-2rings.mat')

% Let n be the number of examples in R^d space
% n (j) is 200 and d (i) is 2
[d, n] = size(x);

% Initalize values
t_max = 1000;
C = 256 / n;
sigma = 0.5;

% Find K
K = RBF(n, x, sigma);

%% 8.2(a) Create a plot which shows how the sample-normalized cost evolves.

% Find sample-normalized cost
cost_array = [];
psi_array = [];
for t = 0 : 10 : t_max
    
    % Find psi
    psi = SSGD(x, y, K, t, C);
    psi_array = [psi_array psi];
    
    % Find cost
    cost = (1 / n) * costFunction(psi, K, y, n, C);
    cost_array = [cost_array cost];
    if(t == 10)
        min_cost = cost;
        psi_svm = psi;
    end
    
    % Find minimum gPsi
    if(cost < min_cost)
        min_cost = cost;
        psi_svm = psi;
    end
end
   
% Plot sample-normalized cost for each binary classifier
    figure(1)
    plot(0 : 10 : t_max, cost_array);
    plotName = sprintf('Sample Normalized Cost vs t');
    title(plotName)
    xlabel('t')
    ylabel('Sample Normalized Cost')
    
 %% 8.2(b) Create a plot which shows how the training CCR evolves.
 
% Find the training CCR
ccr_training = [];
j = 1;
for t = 0 : 10 : t_max
    currentPsi = psi_array(:, j);
    [ccr, yj] = CCR(x, y, currentPsi, n);
    ccr_training = [ccr_training ccr];
    j = j + 1;
end
   
% Plot training ccr
    figure(2)
    plot(0 : 10 : t_max, ccr_training);
    plotName = sprintf('Training CCR vs t');
    title(plotName)
    xlabel('t')
    ylabel('Training CCR')
    
%% 8.2(c) Report the Final training confusion matrix

fprintf('\n Training Confusion matrix \n')
confusionmatTraining = confusionmat(y, yj);
disp(confusionmatTraining)


%% 8.2(d) Visualization of Decision Boundrary [HAVING TROUBLE]
class1 = x(:,y == 1);
class2 = x(:,y == -1);
class1_pred = x(:,yj == 1);
class2_pred = x(:,yj == -1);

% Find the decision boundrary.  Where the decision between a pair of points
% changes.
boundrary = [];
for j = 1 : n-1
    
    if(yj(j) ~= yj(j + 1))
        boundrary = [boundrary x(:, j) x(:, j) ];
    end
end


figure(3)
hold on
scatter(class1(1, :), class1(2,:),'b+')
scatter(class2(1,:), class2(2,:),'gs')
scatter(boundrary(1,:), boundrary(2,:),'mo', 'filled')
title('Training set and decision boundary')
xlabel('Class 1');
ylabel('Class 2');
plotName = sprintf('Visualization of Decision Boundrary');
title(plotName)
legend('Class 1','Class 2', 'Decision Boundrary')
hold off

%% Functions

% Radial Basis Function (RBF) Kernel
function K = RBF(n, x, sigma)

% Find the kernel for each pair of feature vectors
% Let K be a n x n matrix
K = zeros(n);

for j = 1 : n
    for i = 1 : n
        K(i, j) = exp( (- 1 / (2 * sigma^2)) *  ( norm( x(:,i) - x(:, j)) ).^ 2 );
    end
end

end

% Hinge Function
function result = hinge(yj, psi, Kj_ext)

% Must find t, equation given in HW
t = yj * psi' * Kj_ext;

result = max( [0, (1-t)] );
end

% Stochastic Sub-Gradient Descent for binary Kernel SVM
function psi = SSGD(x, y, K, t_max, C)

% Intialize variables
[~, n] = size(x);
psi = zeros(n + 1, 1);
range = 1 : n;

for t = 1 : t_max
    
    % Choose sample index
    j = randsample(range, 1);
    
    % Initialize st
    s_t = 0.256 / t;
    
    % Creating a extended K matrix with zeros
    K_w_zeros = zeros(201);
    K_w_zeros(1 : 200, 1: 200) = K;
    
    % Creating Kj extended with a 1
    Kj_ext = [K(:,j); 1];
    
    % Compute gradients
    v = K_w_zeros * psi;
    
    % Update parameters if true
    if(y(j) * psi' * Kj_ext < 1)
        v = v - n * C * y(j) * Kj_ext;
    end
    
    % Update psi always
    psi = psi - s_t * v;
end
end

% Soft-Margin SVM
function gPsi = costFunction(psi, K, y, n, C)

% Creating a extended K matrix with zeros
K_w_zeros = zeros(201);
K_w_zeros(1 : 200, 1: 200) = K;

% Find f0(psi)
f0 = (1/2) * psi' * K_w_zeros * psi;

% Find fj(psi)
fj = 0;
for j = 1 : n
    
    % Creating Kj extended with a 1
    Kj_ext = [K(:,j); 1];
    temp = C * hinge(y(j), psi, Kj_ext);
    fj = fj + temp;
end

gPsi = f0 + fj;

end

% Decision Rule and CCR
function [ccr, yj] = CCR(x, y, psi, n)

% Initialize variables
yj = zeros(n, 1);

% Find decision rule for the training set
for j = 1 : n
    
    % Get K of xtest [MIGHT NEED TO BE RBF[
    Kxtest = x' * x(:, j);
    Kxtest_ext = [Kxtest; 1];
    yj(j) = sign(psi' * Kxtest_ext);
    
end

% CCR
ccr = (1 / n) * sum(y == yj);
end