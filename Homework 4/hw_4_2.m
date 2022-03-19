%(a)
x = [-1 -1/2 0 1/2 1];
y = [-1 -1/8 0 1/8 1];

n = length(y);

mux = (sum(x,2)) * 1/n;
muy = (sum(y,2)) * 1/n;

X = x - mux * ones(n,1)';
Y = y - muy * ones(n,1)';

Sx = 1/n * X * X';
Sxy = 1/n * X * Y';

wOLS = inv(Sx)*Sxy
bOLS = muy - wOLS'*mux

%(b)
x_cubed = x.^3;
Sx_cubed = 1/n * (x_cubed * x_cubed');

Sxy_cubed = 1/n * x_cubed * Y';

x_squared = x.^2;
mux_squared = (sum(x_squared,2)) * 1/n;
X_squared = x_squared - mux_squared * ones(n,1)';
Sx_squared = 1/n * (x_squared * x_squared');
Sxy_squared = 1/n * X_squared * Y';
