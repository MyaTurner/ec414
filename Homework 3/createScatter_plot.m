function [] = createScatter_plot(dataMatrix, labels,  k)

% Create scatter plot for clusters
for i = 1 : k
    labeledData = dataMatrix((find(labels == k)), :);
    scatter(labeledData(:,1), labeledData(:, 2));
    hold on
end

