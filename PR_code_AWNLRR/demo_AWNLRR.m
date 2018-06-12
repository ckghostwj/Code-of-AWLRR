% The code is written by Jie Wen, 
% if you have any problems, please don't hesitate to contact me: wenjie@hrbeu.edu.cn 
% If you find the code is useful, please cite the following reference:
% J. Wen, B. Zhang, Y. Xu, J. Yang, and N. Han, 
% Adaptive Weighted Nonnegative Low-Rank Representation, 
% Pattern Recognition, 2018.

clear all
clc
clear memory;
name = 'COIL20';
load (name);

fea = fea';
fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
n = length(gnd);
nnClass = length(unique(gnd));  

options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'Binary';
Z = constructW(fea',options);
Z_ini = full(Z);
clear LZ Z Z1 options

% % if you only have cpu do this 
% Ctg = inv(fea'*fea+eye(size(fea,2)));

% % -------- if you have gpu you can accelerate the inverse operation as follows:  ---------- % %
Xg = gpuArray(single(fea));
Ctg = inv(Xg'*Xg+eye(n));
Ctg = double(gather(Ctg));
clear Xg;

lambda1 = 10
lambda2 = 0.001
lambda3 = 0.02
miu = 1e-2;
rho = 1.1;
max_iter = 80;
[Z,S,obj] = AWLRR(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
addpath('Ncut_9');
Z_out = Z;
A = Z_out;
A = A - diag(diag(A));
A = abs(A);
A = (A+A')/2;  
[NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(A,nnClass);
result_label = zeros(size(fea,2),1);
for j = 1:nnClass
    id = find(NcutDiscrete(:,j));
    result_label(id) = j;
end
result = ClusteringMeasure(gnd, result_label);
acc  = result(1)
nmi  = result(2)                                                        