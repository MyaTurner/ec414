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

wOLS = inv(Sx)*Sxy;
bOLS = muy - wOLS'*mux;

%(b)
theta = [x.^3; x.^2];

mu_theta = mean(theta');

theta_centered = theta - mu_theta';

S_theta = 1/n * theta_centered * theta_centered';

S_theta_inv = inv(S_theta);
S_theta_y = 1/n * theta_centered * y';

wpls = S_theta_inv * S_theta_y;

