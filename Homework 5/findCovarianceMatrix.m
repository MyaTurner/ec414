function Sx = findCovarianceMatrix(x)
%FINDCOVARIANCEMATRIX Summary of this function goes here
%   Detailed explanation goes here
n = width(x);

mux = mean(x');

x_centered = x - mux';

Sx = (1/n) * (x_centered * x_centered');
end

