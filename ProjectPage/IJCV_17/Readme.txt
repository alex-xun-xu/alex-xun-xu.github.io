Data release for [1] and [2]

Please cite these works for any use of the data.

####################################
##	Fisher Vector Encoded Dense Trajecotry Features

The released data is for fisher vector encoded feature of HMDB51, UCF101, Olympic Sports and CCV datasets. Please download from http://pan.baidu.com/s/1pLcAweZ at baidu Cloud Disk

Improved trajectory features [3] with default setting are computed for each dataset. Descriptors (HOF, HOG and MBH) are reduced to half of the original dimension by PCA. 256,000 random descriptors are sampled from all datasets upon which Gaussian Mixture Model with 128 centers are learned. The GMM is used to fisher vector encode all videos.

We provide each descriptor after fisher vector encoding (e.g. ./HMDB51/HOF.mat) stored as a matrix. Each row is one video example while each column is one feature dimension.

To use these feature, please load each descriptor and do power followed by L2 normalization (functions can be found in the code section). Normalized descriptors are then concatenated to form final feature. We provide two matlab functions (at the end of this file) to do the normalization and a sample script to load feature in matlab.

%% Sample Matlab Code to Load Feature
load('./HMDB51/HOF.mat','HOF');
load('./HMDB51/HOG.mat','HOG');
load('./HMDB51/MBH.mat','MBH');

alpha = 0.2;
%% Power Normalization
HOF_norm = func_PowerNormalization(HOF,alpha);
%% L2 Normalization
HOF_norm = func_L2Normalization(HOF_norm);

%% Concatenate Decriptors
AllFeat = [HOF_norm ; HOG_norm ; MBH_norm];

####################################
##	Labels, Splits and Word-Vectors

We provide the labels and word-vector representations for each category name in https://pan.baidu.com/s/1cEslaprkH3dWi7A15rZOng. Note, the CCV dataset is multi-labelled, i.e. each video may be associated with multiple labels.

50 random splits for zero-shot evaluation on all 4 datasets are given in https://pan.baidu.com/s/106pqzY-t6UZsLfw_y8S1pg .

Alternative to Baidu Disk, all data can be downloaded from https://drive.google.com/open?id=0B0Ahi0YU7ffLZnNTLUJsLVh4WWs

Reference:
[1] Xu, X., Hospedales, T. and Gong, S., (2017). Transductive Zero-Shot Action Recognition by Word-Vector Embedding. International Journal of Computer Vision, pp.1-25.
[2] Xu, X., Hospedales, T. and Gong, S., (2016). Multi-Task Zero-Shot Action Recognition with Prioritised Data Augmentation. European Conference on Computer Vision.
[3] Wang, H., Oneata, D., Verbeek, J., & Schmid, C. (2015). A robust and efficient video representation for action recognition. International Journal of Computer Vision, 1-20.

####################################
## functions 

%% L2 Normalization
feat_L2 = function_L2Normalization(feat)

feat_L2 = feat./repmat(sqrt(sum(feat.^2,2))+eps,1,size(feat,2));

end

%% Power Normalization
feat_Pow = func_PowerNormalization(feat,alpha)

signs = sign(feat);
feat_Pow=signs.*(abs(feat).^alpha);

end

