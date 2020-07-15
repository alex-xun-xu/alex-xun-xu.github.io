%% Normalize Feature Matrix

function normFeatureMatrix = func_NormalizeFeatureMatrix(FeatureMatrix)

normFeatureMatrix = FeatureMatrix./repmat(sum(FeatureMatrix,2),1,size(FeatureMatrix,2));