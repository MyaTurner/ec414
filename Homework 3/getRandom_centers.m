function randomMu = getRandom_centers(dataMatrix, N)

% Generate N random indices
randomMU_indices = randi([1 150],1, N);

randomMu = dataMatrix(randomMU_indices, :);

end

