function randomMu = getRandom_centers(dataMatrix, N)

% Find x-distance min and max
x_min = min(dataMatrix(:,1));
x_max = max(dataMatrix(:,1));

% Find y-distance min and max
y_min = min(dataMatrix(:,1));
y_max = max(dataMatrix(:,1));

% Generate 10 random centers
randomMU_x = x_min + (x_max - x_min).* rand(N, 1);
randomMU_y = y_min + (y_max - y_min).* rand(N, 1);
randomMu = [randomMU_x randomMU_y];
end

