%% script Test HMDB Zeroshot learning

addpath('../function/');

%% Parameter
perc_TrainingSet = 0.5;
perc_TestingSet = 1 - perc_TrainingSet;
cluster_type = 'vlfeat';
nSample = 1e5;
CodebookSize = 4000;
process = 'org'; % preprocess of dataset: org,sta
FEATURETYPE = 'DenseTrj|HOF|HOG|MBH';
nPCA = 0;
C = 2^1; % Cost parameter for SVR
SelfTraining = 0;   % Indicator if do selftraining
trial = 1;
EmbeddingMethod = 'add'; % add , multiply , combine


%% Do M independent splits
M = 30;
meanAcc = [];
for trial = 1:M
    func_tr_SVR(perc_TrainingSet,cluster_type,nSample,CodebookSize,process,FEATURETYPE,nPCA,C,trial,EmbeddingMethod);
    
    meanAcc(trial) = func_ts_SVR(perc_TrainingSet,cluster_type,nSample,CodebookSize,process,FEATURETYPE,nPCA,C,SelfTraining,trial,EmbeddingMethod);
    
    fprintf('%dth trial acc = %.2f\n',trial,meanAcc(trial)*100);
%     fprintf('mean Acc=%.2f\n',mean(meanAcc)*100);
end

%%
fprintf('\n\nZero-Shot Average Accuracy for HMDB Dataset: %.1f +- %.1f\n\n',100*mean(meanAcc),100*std(meanAcc));
fprintf('Corresponds to Method: NN in Table.1 ''Xu Et Al. Semantic Embedding Space for Zero-Shot Action Recognition, ICIP 15''\n\n');

