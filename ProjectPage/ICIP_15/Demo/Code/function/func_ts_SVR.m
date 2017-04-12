%% Test the Support Vector Regression from Dense Trajectory BoW to WordVector on ZSL
%
%   project test samples into word vector space by SVR and do Nearest
%   Neighbour matching to classify test samples
%
%   varargin -
%   varargin{1} - perc_TrainingSet,
%   varargin{2} - cluster_type - the technique used for Kmeans,
%   varargin{3} - nSample - the number of samples for generating codebook,
%   varargin{4} - CodebookSize - the number of centers for Kmeans, varargin
%   varargin{5} - preprocess of dataset: org (original video) or sta (stabilized video),
%   varargin{6] - Descriptor, descriptors used for construct visual feature
%   Dense Trajectory, HOG, HOF and MBH
%   varargin{7} - do pca on input data, if 0 don't do pca if
%   nonzero, do pca and take the first # varargin{7} dims as the process training data. rbdchisq kernel is applied
%   Self-training is applied to drift prototypes
%   varargin{8} - C parameter for Support Vector Regression
%   varargin{9} - SelfTraining - indicator for doing self training
%   varargin{10} - trial - the number of trials 1 to 30
%   varargin{11} - EmbeddingMethod - method to construct category embedding

function meanAcc = func_ts_SVR(varargin)


addpath('../libsvm_3_16/matlab/');


%% Parse Input
if isempty(varargin)
    Para.perc_TrainingSet = 0.5;
    Para.perc_TestingSet = 1 - Para.perc_TrainingSet;
    Para.cluster_type = 'vlfeat';
    Para.nSample = 1e5;
    Para.CodebookSize = 4000;
    Para.process = 'org'; % preprocess of dataset: org,sta
    Para.Descriptor = 'DenseTrj|HOF|HOG|MBH';
    Para.nPCA = 0;
    Para.C = 2^1;
    Para.SelfTraining = 1;   % Indicator if do selftraining
    Para.trial = 1;
    Para.EmbeddingMethod = 'add';
else
    if nargin >=1
        Para.perc_TrainingSet = varargin{1};
    else
        Para.perc_TrainingSet = 0.5; % Training set percentage
    end
    Para.perc_TestingSet = 1 - Para.perc_TrainingSet;
    
    if nargin >=2
        Para.cluster_type = varargin{2};
    else
        Para.cluster_type = 'vlfeat';
    end
    
    if nargin >=3
        Para.nSample = varargin{3};
    else
        Para.nSample = 1e5;
    end
    
    if nargin >=4
        Para.CodebookSize = varargin{4};
    else
        Para.CodebookSize = '2048';
    end
    
    if nargin >=5
        Para.process = varargin{5};
    else
        Para.process = 'sta';
    end
    
    if nargin >=6
        Para.Descriptor = varargin{6};
    else
        
        Para.Descriptor = 'DenseTrj|HOF|HOG|MBH';
    end
    
    if nargin >=7
        Para.nPCA = varargin{7};
    else
        Para.nPCA = 0;
    end
    
    if nargin >=8
        Para.C = varargin{8};
    else
        Para.C = 2;
    end
    
    if nargin >=9
        Para.SelfTraining = varargin{9};
    else
        Para.SelfTraining = 0;
    end
    
    if nargin >=10
        Para.trial = varargin{10};
    else
        Para.trial = 1;
    end
    
    if nargin >=11
        Para.EmbeddingMethod = varargin{11};
    else
        Para.EmbeddingMethod = 'add';
    end
end

%% Function Internal Parameters
Para.LabelVector = 'label'; % which kind of word vector strategy is used for class representation: label (the single label), add (sum up all descriptions)
Para.DataPath = '../../Data/';
Para.WordVecPath = '../../Data/HMDB_WordVector/';
Para.FEATURE = 'DenseTrj'; % DETECTOR type: STIP, DenseTrj
Para.norm_flag = 1;   % normalization strategy: org,histnorm,zscore

%%% Determine which descriptors are included
ind = 1;
rest = Para.Descriptor;
while true
    [Para.DescriptorList{ind},rest] = strtok(rest,'|');
    if isempty(rest)
        break;
    end
    ind = ind+1;
end

%% Load Video Groundtruth Label
temp = load(fullfile(Para.DataPath,'HMDB_Datasplit','PerVideoLabel.mat'));
Para.ClassNoPerVideo = temp.ClassNoPerVideo;

%% Load ZSL Category Split
temp = load(sprintf(fullfile(Para.DataPath,'HMDB_Datasplit','DatasetSplit_tr-%.1f_ts-%.1f_t-%d.mat'),Para.perc_TrainingSet,Para.perc_TestingSet,Para.trial));
Para.idx_TrainingSet = sort(temp.idx_TrainingSet,'ascend');
Para.idx_TestingSet = sort(temp.idx_TestingSet,'ascend');

%% Load Label Word Vector Representation
temp = load(sprintf(fullfile(Para.WordVecPath,'ClassLabelPhraseDict_mth-%s.mat'),Para.EmbeddingMethod));
Data.ClassLabelsPhrase = temp.ClassLabelsPhrase;
Data.phrasevec_mat = temp.phrasevec_mat;

%% Prepare Training Data
Data.tr_FeatureMat = [];
Data.tr_LabelVec = [];
Data.ts_LabelVec = [];
Data.ts_ClassNo = [];
Data.ts_sample_ind = [];   % test sample index
Data.tr_sample_ind = zeros(size(Para.ClassNoPerVideo,1),1);   % train sample index

%%% Testing Samples
clear temp;
for c_ts = 1:length(Para.idx_TestingSet)
%     temp.currentClassName = Data.ClassLabelsPhrase{Para.idx_TestingSet(c_ts)};
    temp.ts_sample_class_ind = ismember(Para.ClassNoPerVideo(:,1),Para.idx_TestingSet(c_ts));
    Data.ts_LabelVec = [Data.ts_LabelVec ; repmat(Data.phrasevec_mat(Para.idx_TestingSet(c_ts),:),sum(temp.ts_sample_class_ind),1)];
    Data.ts_ClassNo = [Data.ts_ClassNo ; repmat(Para.idx_TestingSet(c_ts),sum(temp.ts_sample_class_ind),1)];
end

%%% Training Samples
clear temp;
for c_tr = 1:length(Para.idx_TrainingSet)
%     temp.currentClassName = Data.ClassLabelsPhrase{Para.idx_TrainingSet(c_tr)};
    temp.tr_sample_class_ind = ismember(Para.ClassNoPerVideo(:,1),Para.idx_TrainingSet(c_tr));
    Data.tr_LabelVec = [Data.tr_LabelVec ; repmat(Data.phrasevec_mat(Para.idx_TrainingSet(c_tr),:),sum(temp.tr_sample_class_ind),1)];
    Data.tr_sample_ind = Data.tr_sample_ind + temp.tr_sample_class_ind;
end

%% Testing Support Vector Regression model for each dimension
Para.KernelType = 'rbfchisq';
%%% Load SVR models
Para.regression_path = sprintf(fullfile(Para.DataPath,'HMDB_SVR'),Para.FEATURE);
load(sprintf(fullfile(Para.regression_path,'SVR_c-%d_t-%d_embed-%s_singlecodebook.mat'),Para.C,Para.trial,Para.EmbeddingMethod),'model','tr_LabelVec_hat','ts_LabelVec_hat');

%% Knn to predict final labels
%%% Generate Prototypes
Data.Prototype = Data.phrasevec_mat(Para.idx_TestingSet,:);
temp.SS = sum(Data.Prototype.^2,2);
temp.label_k = sqrt(size(Data.Prototype,2)./temp.SS);
Data.Prototype = repmat(temp.label_k,1,size(Data.Prototype,2)) .* Data.Prototype;

%%% Normalize ts Label Vectors
temp.SS = sum(ts_LabelVec_hat.^2,2);
temp.label_k = sqrt(size(ts_LabelVec_hat,2)./temp.SS);
ts_LabelVec_hat = repmat(temp.label_k,1,size(ts_LabelVec_hat,2)) .* ts_LabelVec_hat;

if Para.SelfTraining
    %% Self-training
    for K = 1:200;
        stPrototype = func_SelfTraining(Data.Prototype, ts_LabelVec_hat, K);
        
        stPrototype = func_SelfTraining(stPrototype, ts_LabelVec_hat, K);
        
        
        %%% Predict labels
        predict_ClassNo = knnsearch(stPrototype,ts_LabelVec_hat,'Distance','cosine');
        
        %%% Calculate Average Precision for each Class
        Accuracy = zeros(1,length(Para.idx_TestingSet));
        for c_ts = 1:length(Para.idx_TestingSet)
            
            currentClass = Para.idx_TestingSet(c_ts);
            currentClass_SampleIndex = Data.ts_ClassNo==currentClass;
            currentClass_Predict = predict_ClassNo(currentClass_SampleIndex);   % predicted class no
            Accuracy(c_ts) = sum(currentClass_Predict == c_ts)/length(currentClass_Predict);
            
        end
        
        meanAcc(K) = mean(Accuracy);
    end
    
else
    %% Without SelfTraining
    predict_ClassNo = knnsearch(Data.Prototype,ts_LabelVec_hat,'Distance','cosine');
    
    %%% Calculate Average Precision for each Class
    for c_ts = 1:length(Para.idx_TestingSet)
        
        currentClass = Para.idx_TestingSet(c_ts);
        currentClass_SampleIndex = Data.ts_ClassNo==currentClass;
        currentClass_Predict = predict_ClassNo(currentClass_SampleIndex);   % predicted class no
        Accuracy(1,c_ts) = sum(currentClass_Predict == c_ts)/length(currentClass_Predict);
        
    end
    
    meanAcc = mean(Accuracy);
end

meanAcc = max(meanAcc);

function [stPrototypes] = func_SelfTraining(Prototype, LabelVector, K)
%% Do self-training on prototypes
%
%   Prototypes moves towards the K

IDX = knnsearch(LabelVector,Prototype,'Distance','euclidean','K',K);

for p_i = 1:size(Prototype,1)
    stPrototypes(p_i,:) = mean(LabelVector(IDX(p_i,:),:),1);
end

function stPrototypes = func_SelfTraining_Median(Prototype, LabelVector, K)
%% Do median self-training on prototypes
%
%   Prototypes moves towards the K

[IDX,D] = knnsearch(LabelVector,Prototype,'Distance','euclidean','K',K);

for p_i = 1:size(Prototype,1)
    idx = round(K/2);
    stPrototypes(p_i,:) = LabelVector(IDX(p_i,idx),:);
end
