function [ randomMu, usedIndices ]= getRandom_centers(dataMatrix, N, alreadyUsed)

num_datapoints = length(dataMatrix);

if (~isempty(alreadyUsed))
    
    % Generate N random indices
    validVals = setdiff(1 : num_datapoints, alreadyUsed);
    randomMU_indices = validVals( randi(length(validVals), 1, N)); 
else
   % Generate N random indices
    randomMU_indices = randi([1 num_datapoints],1, N);
end

randomMu = dataMatrix(randomMU_indices, :);
usedIndices = randomMU_indices;

end

