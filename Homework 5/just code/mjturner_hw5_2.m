function [lambda_top5, k_] = mjturner_hw5_2()
%% Q5.2
%% Load AT&T Face dataset
    img_size = [112,92];   % image size (rows,columns)
    % Load the AT&T Face data set using load_faces()
    %%%%% TODO
    
    faces = load_faces();
    X = faces';
    [d,n] = size(X);

    %% Compute mean face and the covariance matrix of faces
    [Sx_faces, x_centered, mux] = findCovarianceMatrix(X);
    
    
    %% Compute the eigenvalue decomposition of the covariance matrix
    [V,D] = eig(x_centered' * x_centered);
    
    %% Sort the eigenvalues and their corresponding eigenvectors construct the U and Lambda matrices
    %%%%% TODO
    % Want descending order the eigenvalues
    eigenvalues = diag(D);
    [eigenvalues_sorted, orig_index] = sort(eigenvalues, 'descend');
    D_sorted = diag(eigenvalues_sorted);
    V_sorted = V(:, orig_index);
  
    % Get orthonormal version of sorted eigenvectors to find U
    % Do this by finding the singular value decomposition (SVD)
    % Note A = X centered to the mean
    U = zeros(d,d);
    for i = 1 : n
        U(:, i) = (1 / sqrt(eigenvalues(i)) ) * x_centered * V_sorted(:, i); 
    end
    
    
    % Finding lambda matrix by dual PCA
    lambda_mat = zeros(d,d);
    lambda_mat(1:n,1:n) = (1/n) * D_sorted;
    
    %% Compute the principal components: Y
    %%%%% TODO
    
    % Note Wpca = U
    w_pca = U;
    
    % Find y_pca
    y_pca = w_pca' * (X - mux');
    

%% Q5.2 a) Visualize the loaded images and the mean face image
    figure(1)
    sgtitle('Data Visualization')
    
    % Visualize image number 120 in the dataset
    % practice using subplots for later parts
    subplot(1,2,1)
    %%%%% TODO
    imshow(uint8(reshape(X(:,120)', img_size)));
    title('image #120 in the dataset')
    
    % Visualize the mean face image
    subplot(1,2,2)
    %%%%% TODO
    imshow(uint8(reshape(mux', img_size)));
    title('mean face of the dataset')
    
%% Q5.2 b) Analysing computed eigenvalues
    warning('off')
    
    % Report the top 5 eigenvalues
    top_eigenvalues = eigenvalues(1:5);
    fprintf('The first five eigenvalues are: [');
    fprintf('%.3g ', top_eigenvalues);
    fprintf(']');
    fprintf('\n');
    
    % Plot the eigenvalues in from largest to smallest
    d = 450;
    k = 1: d;
    figure(2)
    sgtitle('Eigenvalues from largest to smallest')

    % Plot the eigenvalue number k against k
    subplot(1,2,1)
    % Need to use the larger lambda matrix.  D sorted only goes to index 400
    lambdas = diag(lambda_mat);
    plot(k, lambdas(1 : d));
    title('Eigenvalues v k');
    xlabel('k');
    ylabel('lambda');
    
    % Plot the sum of top k eigenvalues, expressed as a fraction of the sum of all eigenvalues, against k
    %%%%% TODO: Compute eigen fractions
    pk = zeros(d, 1);
    for i = k
        pk(i) = sum(lambdas(1 : i)) / sum(lambdas(1 : d));
    end
    
    % Round pk
    pk = round(pk, 2);
    
    subplot(1,2,2)
    %%%%% TODO
    plot(k, pk);
    title('Fraction of Variance');
    xlabel('k');
    ylabel('top k principal components');
    
    % find & report k for which the eigen fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    %%%%% TODO (Hint: ismember())
    [~, small_vals_locations] = ismember(ef, pk);
    small_vals_k = k(small_vals_locations);
    fprintf('The smallest values of k for which pk >= 0.51, 0.75, 0.90, 0.95 and 0.99 are: [');
    fprintf('%.3g ', small_vals_k);
    fprintf(']');
    fprintf('\n');
    
%% Q5.2 c) Approximating an image using eigen faces
%     test_img_idx = 43;
%     test_img = X(test_img_idx,:);    
%     % Compute eigenface coefficients
%     %%%% TODO
%     
%     K = [0,1,2,k_,400,d];
%     % add eigen faces weighted by eigen face coefficients to the mean face
%     % for each K value
%     % 0 corresponds to adding nothing to the mean face
% 
%     % visulize and plot in a single figure using subplots the resulating image approximations obtained by adding eigen faces to the mean face.
% 
%     %%%% TODO 
%     
%     figure(3)
%     sgtitle('Approximating original image by adding eigen faces')

%% Q5.2 d) Principal components capture different image characteristics
%% Loading and pre-processing MNIST Data-set
    % Data Prameters
    q = 5;                  % number of percentile points
    noi = 3;                % Number of interest
    img_size = [16, 16];
    
    % load mnist into workspace
    mnist = load('mnist256.mat').mnist;
    label = mnist(:,1);
    X = mnist(:,(2:end));
    num_idx = (label == noi);
    X = X(num_idx,:);
    X = X';
    [d,n] = size(X);
    
    %% Compute the mean face and the covariance matrix
    % compute X_tilde
    %%%%% TODO
    [Sx_faces, x_centered, mux] = findCovarianceMatrix(X);
    
    % Compute covariance using X_tilde
    %%%%% TODO
    
    %% Compute the eigenvalue decomposition
    %%%%% TODO
    [V,D] = eig(x_centered' * x_centered);
    
    %% Sort the eigenvalues and their corresponding eigenvectors in the order of decreasing eigenvalues.
    %%%%% TODO
    eigenvalues = diag(D);
    [eigenvalues_sorted, orig_index] = sort(eigenvalues, 'descend');
    D_sorted = diag(eigenvalues_sorted);
    V_sorted = V(:, orig_index);
    
    U = zeros(d,d);
    for i = 1 : n
        U(:, i) = (1 / sqrt(eigenvalues(i)) ) * x_centered * V_sorted(:, i); 
    end
    
   
    %% Compute principal components
    %%%%% TODO
    
    % Note Wpca = U
    w_pca = U;
    
    % Find y_pca
    y_pca = w_pca' * (X - mux');
    
    %% Computing the first 2 pricipal components
    %%%%% TODO
    first_pca = y_pca(:, 1);
    second_pca = y_pca(:, 2);

    % finding percentile points
    percentile_vals = [5, 25, 50, 75, 95];
    %%%%% TODO (Hint: Use the provided fucntion - percentile_points())
    first_percentiles = percentile_values(first_pca, percentile_vals);
    second_percentiles = percentile_values(second_pca, percentile_vals);
    
    % Finding the cartesian product of percentile points to find grid corners
    %%%%% TODO
    [x1,x2] = meshgrid(first_percentiles, second_percentiles);
    % List matrices (grid) as a set of coordinates (x1, x2)
    pg = [x1(:) x2(:)];


    
    %% Find images whose PCA coordinates are closest to the grid coordinates in terms of euclidean distance
    % This is easily done with dsearchn (found by google search)
    %%%%% TODO
    pca_points = [first_pca, second_pca];
    closest_coordinates = dsearchn(pca_points, pg);

    %% Visualize loaded images
    % random image in dataset
    figure(4)
    sgtitle('Data Visualization')

    % Visualize the 100th image
    subplot(1,2,1)
    %%%%% TODO
    % Note unint8 not needed for this dataset to be visuliazed
    imshow((reshape(X(:,120)', img_size)));
    title('image #120 in the dataset of numeral 3')
    
    
    % Mean face image
    subplot(1,2,2)
    %%%%% TODO
    imshow((reshape(mux', img_size)));
    title('mean face of the dataset')
    

    
    %% Image projections onto principal components and their corresponding features
    
    figure(5)    
    hold on
    grid on
    
    % Plotting the principal component 1 vs principal component 2. Draw the
    % grid formed by the percentile points and highlight the image points that are closest to the 
    % percentile grid corners
    
    %%%%% TODO (hint: Use xticks and yticks)
    
    x = real(first_percentiles);
    y = real(second_percentiles);

    xlabel('Principal component 1')
    ylabel('Principal component 2')
    title('Image points closest to percentile grid corners')
    % Not working !! can not figure out why.
   % xticks(sort(x));
   % yticks(sort(y))
    scatter(first_pca(closest_coordinates),second_pca(closest_coordinates),'or','filled');
    hold off
    
    figure(6)
    sgtitle('Images closest to percentile grid corners')
    hold on
    % Plot the images whose PCA coordinates are closest to the percentile grid 
    % corners. Use subplot to put all images in a single figure in a grid.
    
    %%%%% TODO
    % creating a for loop to create subplots
    for i = 1 : length(closest_coordinates)
        
        subplot(5 , 5 ,i)
        % using method used to plot image 120 in dataset
        imshow(reshape(X(:,closest_coordinates(i))', img_size));
        plot_name = sprintf('Image %d', closest_coordinates(i));
        title(plot_name);
    end
    
    hold off    
end