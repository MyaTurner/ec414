%% Homework 8.3

% Load dataset
load('kernel-kmeans-2rings.mat')

% Let n be the number of examples in R^d space
% n (j) is 1000 and d (i) is 2
[d, n] = size(data);


% Initalize values
sigma = 0.4;
k = 2;


%% Functions

% Kernel k-means Clustering Algorithm
function kernel_kmeans(data, k)

end

% Radial Basis Function (RBF) Kernel % Implemented wrong, need to [FIX]
function K = RBF(n, x, sigma)

% Find the kernel for each pair of feature vectors
% Let K be a n x n matrix
K = zeros(n);
X_transposed = x';
X = x;

% for i = 1 : n
%     j = i;
%     K(i,j) = exp( (- 1 / (2 * sigma^2)) * norm(X_transposed(i,:) - X(:,j ).^ 2) );
% end

K = x' * x;
end