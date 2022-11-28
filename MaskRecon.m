
addpath(genpath('/home/work/SemiPark/RECON'))
addpath(genpath('/home/work/SemiPark/RECON/Utils'))
addpath(genpath('/home/work/SemiPark/semi_idx'))

[DIR] = DIRset();
if (DIR.NumFile == 1)
    Filename = DIR.NAME;
else
    Filename = DIR.NAME{File};
end
DataLoad_20220324
TrajSet_a_UTE
ReconParam_a_UTE
[Radius] = RampUpCorr_a_UTE(Info);

disp('finish load data')
close all
clearvars -except Gx Gy Gz fid Radius matrixsize 

coilsen_real = single(niftiread('coilsen_real.nii'));
coilsen_imag = single(niftiread('coilsen_imag.nii'));
coilsen = complex(coilsen_real, coilsen_imag);
clear coilsen_real coilsen_imag
coilsen = flip(coilsen,1);
coilsen = flip(coilsen,2);
coilsen = flip(coilsen,3);
disp('finish load coil sensitivity')

k_angle = zeros(size(fid,1),3,size(fid,3));
for i = 1:size(fid,3)
    k_angle(:,1,i) = Gx(i)*Radius;
    k_angle(:,2,i) = Gy(i)*Radius;
    k_angle(:,3,i) = Gz(i)*Radius;
end
clear Gx Gy Gz

load('idx1_8.mat');load('idx2_8.mat');load('idx3_8.mat');load('idx4_8.mat');load('idx5_8.mat')
index{1} = idx1_8;index{2} = idx2_8;index{3} = idx3_8;index{4} = idx4_8;index{5} = idx5_8;
clear idx1_8 idx2_8 idx3_8 idx4_8 idx5_8

nsamps = size(fid,1);
ncoils = size(fid,2);
nviews = size(fid,3);
% nframes = size(index,2);
nframes=1;
for i = 1:nframes
    nsampviews{i} = size(index{i},1);
end

%ram-lak filter for 3d k-space
% w = single((Radius).^2);
% Radius = single(Radius);
% ww = repmat(w,[1 ncoils nviews]);

for i = 1:nframes
    kdatau{i} = fid(:,:,index{i});
    ku{i} = k_angle(:,:,index{i});
end
clear fid k_angle
disp('finish making 5 phase')
disp('setup start')

for nf = 1:nframes
    k = ku{nf};
    k1 = k(:,1,:);
    k2 = k(:,2,:);
    k3 = k(:,3,:);
    ktraj = [k1(:),k2(:),k3(:)];
    [kerneldistance{nf}, xyz_index{nf}, index_smth2{nf}, mask{nf}, win_3d] = setup3d(ktraj,matrixsize);
end
clear k k1 k2 k3 ktraj

g = gpuDevice();
reset(g);
coilsen = gpuArray(coilsen);

kdatau_gpu = gpuArray(reshape(permute(kdatau{1},[1 3 2]),[],ncoils));
kerneldistance_gpu = gpuArray(repmat(kerneldistance{1},[1 ncoils]));
xyz_index_gpu = gpuArray(double(xyz_index{1}));
win_3d_gpu = gpuArray(win_3d);
mask_gpu = gpuArray(mask{1});
index_smth2_gpu = gpuArray(index_smth2{1});
tic
[IMGF] = MaskForwardGridding2(kdatau_gpu, coilsen, kerneldistance_gpu, xyz_index_gpu, matrixsize, index_smth2_gpu, win_3d_gpu, mask_gpu);
toc
figure(1); imagesc(abs(squeeze(IMGF(:,440,:))));
nii = make_nii(squeeze(abs(IMGF(221:660,221:660,221:660)))); save_nii(nii,'gpu_mask.nii');