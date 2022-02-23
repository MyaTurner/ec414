function XDist = euclideanDistance(, MU_init)

%Compute the euclidean distance for each row in given matrix from mu
for i = 

for test_row = 1 : Ntest
    for train_row = 1 : Ntrain
        XDist(test_row, train_row) = sqrt( sum ((Xtest(test_row, :) - Xtrain(train_row, :)).^2) );
    end
end
end

