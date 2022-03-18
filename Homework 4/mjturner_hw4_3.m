%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 414 (Ishwar) Spring 2022
% HW 4
% Mya Turner: mjturner@bu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;
rng('default')  % For reproducibility of data and results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(a)
% Generate and plot the data points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];

% Generate dataset (i), (ii), (iii), (iv)

lambda1 = [1 1 1 0.25];
lambda2 = [0.25 0.25 0.25 1];
theta = [0 1 2 1];

for i = 1:4
    [X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1(i), lambda2(i), theta(i) * pi/6);
    
    % Scatter plot of the generated dataset (i)
    X1 = X(:, Y==1);
    X2 = X(:, Y==2);

    figure(1);subplot(2,2,i);
    scatter(X1(1,:),X1(2,:),'o','fill','b');
    grid;axis equal;hold on;
    xlabel('x_1');ylabel('x_2');
    title(['\theta = ',num2str(theta(i)),'\times \pi/6']);
    scatter(X2(1,:),X2(2,:),'^','fill','r');
    axis equal;
end

% Using dataset (ii) for the rest of the questions
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1(2), lambda2(2), theta(2) * pi/6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(b)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each phi = 0 to pi in steps of pi/48 compute the signal power, noise 
% power, and snr along direction phi and plot them against phi 

phi_array = 0:pi/48:pi;
signal_power_array = zeros(1,length(phi_array));
noise_power_array = zeros(1,length(phi_array));
snr_array = zeros(1,length(phi_array));
for i=1:1:length(phi_array)
    [signal, noise, snr] = signal_noise_snr(X, Y, phi_array(i), false);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = signal;
    noise_power_array(i) = noise;
    snr_array(i) = snr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to create plots of signal power versus phi, noise
% power versus phi, and snr versus phi and to locate the values of phi
% where the signal power is maximized, the noise power is minimized, and
% the snr is maximized
%
figure(2);
scatter(phi_array,signal_power_array)
xlabel('Phi');
ylabel('Signal Power');
title('Signal Power v Phi');

figure(3);
scatter(phi_array,noise_power_array)
xlabel('Phi');
ylabel('Noise Power');
title('Noise Power v Phi');

figure(4);
scatter(phi_array,snr_array)
xlabel('Phi');
ylabel('SNR');
title('SNR v Phi');

maxSignal = max(signal_power_array);
maxSignal_phi = phi_array(find(signal_power_array == maxSignal));
fprintf('Phi that maximizes signal: %.3f\n', maxSignal_phi);

minNoise = min(noise_power_array);
minNoise_phi = phi_array(find(noise_power_array == minNoise));
fprintf('Phi that minimizes noise: %.3f\n', minNoise_phi);

maxSNR = max(snr_array);
maxSNR_phi = phi_array(find(snr_array == maxSNR));
fprintf('Phi that maximizes SNR: %.3f\n\n', maxSNR_phi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For phi = 0, pi/6, and pi/3, generate plots of estimated class 1 and 
% class 2 densities of the projections of the feature vectors along 
% direction phi. To do this, set phi to the desired value, set 
% want_class_density_plots = true; 
% and then invoke the function: 
% signal_noise_snr(X, Y, phi, want_class_density_plots);
% Insert your script here 
% ...
phi_array = [0 1 2];
[f, xi] = ksdensity(X');
for i=1:1:length(phi_array)
    [signal, noise, snr] = signal_noise_snr(X, Y, phi_array(i), true);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = signal;
    noise_power_array(i) = noise;
    snr_array(i) = snr;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the LDA solution by writing and invoking a function named LDA 

w_LDA = LDA(X,Y);

% See below for the LDA function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to create a scatter plot and overlay the LDA vector and the 
% difference between the class means. Use can use Matlab's quiver function 
% to do this.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(d)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create CCR vs b plot

X_project = w_LDA' * X;
X_project_sorted = sort(X_project);
b_array = X_project_sorted * (diag(ones(1,n))+ diag(ones(1,n-1),-1)) / 2;
b_array = b_array(1:(n-1));
ccr_array = zeros(1,n-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise: decode what the last 6 lines of code are doing and why
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:(n-1)
    ccr_array(i) = compute_ccr(X, Y, w_LDA, b_array(i));
end

% See below for the compute_ccr function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to plote CCR as a function of b and determine the value of b
% which maximizes the CCR.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complete the following 4 functions defined below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [X, Y] = two_2D_Gaussians(n1,n2,mu1,mu2,lambda1,lambda2,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function should generate a labeled dataset of 2D data points drawn 
% independently from 2 Gaussian distributions with the same covariance 
% matrix but different mean vectors
u1 = [cos(theta); sin(theta)];
u2 = [sin(theta); -cos(theta)];
U = [u1 u2];
lambdaDiagonal_matrix = [lambda1 0; 0 lambda2];
S_xavg = U * lambdaDiagonal_matrix * U';

X1 = mvnrnd(mu1,S_xavg,n1);
X2 = mvnrnd(mu2, S_xavg, n2);

X = [X1 ;X2]';
Y = ones(1, n1 + n2);
Y(:, n1 + 1 : end) = 2;

end

function [signal, noise, snr] = signal_noise_snr(X, Y, phi, want_class_density_plots)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to project data along direction phi and then comput the
% resulting signal power, noise power, and snr 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Defining each class
X1 = X(:, find(Y==1));
X2 = X(:, find(Y==2));
n1 = width(X1);
n2 = width(X2);
n = n1 + n2;

% Note w is nothing but the direction of projection!
w = [cos(phi); sin(phi)];

% Step 1) Find empirical means for each class
mu1 = mean(X1');
mu2 = mean(X2');

% Step 2) Find empirical probablities of each class
p1 = n1/n;
p2 = n2/n;

% Step 3) Find covariance matrices
% Need to get centered means
X1_centered = zeros(2, n1);
for i = 1: n1
    X1_centered(:, i) = X1(:, i) - mu1';
end

X2_centered = zeros(2, n2);
for i = 1: n1
    X2_centered(:, i) = X2(:, i) - mu2';
end

sigma1 = 1/n1 * X1_centered * X1_centered';
sigma2 = 1/n2 * X2_centered * X2_centered';

% Step 4) Find average covariance matrices
sigmaAvg = p1 * sigma1 + p2 * sigma2;

% Find means projected onto direction phi
muz1 = w' * mu1';
muz2 = w' * mu2';

% Find signal (euclidean distance between means projected to phi)
signal = (muz2 - muz1).^2;

% Project variances onto direction phi
sigmaz1 = w' * sigma1 * w;
sigmaz2 = w' * sigma2 * w;

% Find signal (euclidean distance between means projected to phi)
noise = n1/n * sigmaz1 + n2/n * sigmaz2;

% Find SNR (Signal to Noise Ratio)
snr = ((w' * (mu2 - mu1)').^2) /  (w' * sigmaAvg * w );

% ...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To generate plots of estimated class 1 and class 2 densities of the 
% projections of the feature vectors along direction phi, set:
% want_class_density_plots = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if want_class_density_plots == true
    % Plot density estimates for both classes along chosen direction phi
    figure();
    [pdf1,z1] = ksdensity(w' * X1);
    plot(pdf1,z1)
    hold on;
    [pdf2,z2] = ksdensity(w' * X2);
    plot(pdf2,z2)
    grid on;
    hold off;
    legend('Class 1', 'Class 2')
    xlabel('projected value')
    ylabel('density estimate')
    title(['Estimated class density estimates of data projected along \phi =',num2str(phi),'\times \pi/6']);
end

end

function w_LDA = LDA(X, Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to compute and return the LDA solution
% Defining each class
X1 = X(:, find(Y==1));
X2 = X(:, find(Y==2));
n1 = width(X1);
n2 = width(X2);
n = n1 + n2;

% Need to find wlda 
% Step 1) Find empirical means for each class
mu1 = mean(X1');
mu2 = mean(X2');

% Step 2) Find empirical probablities of each class
p1 = n1/n;
p2 = n2/n;

% Step 3) Find covariance matrices
% Need to get centered means
X1_centered = zeros(2, n1);
for i = 1: n1
    X1_centered(:, i) = X1(:, i) - mu1';
end

X2_centered = zeros(2, n2);
for i = 1: n1
    X2_centered(:, i) = X2(:, i) - mu2';
end

sigma1 = 1/n1 * X1_centered * X1_centered';
sigma2 = 1/n2 * X2_centered * X2_centered';

% Step 4) Find average covariance matrices
sigmaAvg = p1 * sigma1 + p2 * sigma2;
sigmaAvg_inverse = inv(sigmaAvg);

% Step 5) Find wlda
wlda = sigmaAvg_inverse * (mu2 - mu1)';

% Find means projected onto wlda
muz1 = wlda' * mu1';
muz2 = wlda' * mu2';

% Project means onto direction phi
directionPhi = [cos(phi);sin(phi)];
muz1_phi = directionPhi' * muz1';
muz2_phi = directionPhi' * muz2';

% Find signal (euclidean distance between means projected to phi)
signal = (muz2_phi - muz1_phi).^2;

% Find covariances projected onto wlda
sigmaz1 = wlda' * sigma1 * wlda;
sigmaz2 = wlda' * sigma2 * wlda;

% Project variances onto direction phi
sigmaz1_phi = directionPhi' * sigmaz1;
sigmaz2_phi = directionPhi' * sigmaz2;

% Find signal (euclidean distance between means projected to phi)
noise = n1/n * sigmaz1_phi + n2/n * sigmaz2_phi;

% Find SNR (Signal to Noise Ratio)
snr = ((wlda' * (mu2 - mu1)').^2) /  (wlda' * sigmaAvg_inverse * wlda );

% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

function ccr = compute_ccr(X, Y, w_LDA, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to compute the CCR for the given labeled dataset
% (X,Y) when you classify the feature vectors in X using w_LDA and b
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end