%5.1(a)
x = [1 1 0 2; 3 7 5 5];
Sx = findCovarianceMatrix(x);
[V,D] = eig(Sx);

%5.1(b)
r = [-1 2 -1 0 2]';
s = [1 -1 -1 1 1]';
X = [3*r,r,-1*r,-3*r,3*s,2*s,s,-6*s];
[d,n] = size(X)
Sx = findCovarianceMatrix(X)

