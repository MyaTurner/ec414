%% Homework 8.3

% Load dataset
load('kernel-kmeans-2rings.mat')

% Let n be the number of examples in R^d space
% n (j) is 1000 and d (i) is 2
X = data';
[d n] = size(X);
e = eye(n);

% Initalize values
sigma = 0.4;
k = 2;

% Find K
K = RBF(n, X, sigma);

% Run kernel k means
kernel_kmeans(X, k, K, n, e)

%% Functions

% Kernel k-means Clustering Algorithm
function kernel_kmeans(X, k, K, n, e)

% Initialize variables
y = zeros(n, 1);
mu = zeros(n, k);

% Set centers
for l = 1 : k
    mul = rand(n,1);
    
    % Normalize centers
    mu(:, l) = mul / norm(mul);
    
end


% Set stopping condition
STOP = false;
iteration = 0;
while(~STOP)
    
    iteration = iteration + 1;
    
    % Update labels
    % For each feature vector
    for j = 1 : n
        
        cluster = zeros(1, k);
        
        for l = 1 : k
            
            mul = mu(:, l);
            
            % Find best cluster (should be 1 or 2)
            cluster(l) = (e(:, j) - mul)' * K * (e(:, j) - mul);
        end
        
        [~, y(j)] = min(cluster);
        
    end
    
    % Update cluster-mean coefficient vectors
     for l  = 1 : k
         
         nl = sum(y == 1);
         
         if (nl ~= 0)
             summation = 0;
             
             % For each feature vector in cluster k
             for j = 1 : n
                 summation = summation + (e(:,j) * (y(j) == l));
             end
             
             mu(:, l) = (1/nl) * summation;
         else
             mu(:, l) = zeros(n, 1);
             
         end

     end
     
     % Stopping condition
     if(iteration > 50)
         STOP = true;
     end
end

     figure(1)
     gscatter(X(1,:),X(2,:),y)
     title('Kernel k-means clustering')
     xlabel('Class 1')
     ylabel('Class 2')
end

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