% clear; clc; close all;
addpath(genpath('/home/work/SemiPark/RECON'))
addpath(genpath('/home/work/SemiPark/RECON/Utils'))
addpath(genpath('/home/work/SemiPark/semi_idx'))
% 
coilsen_real = niftiread('coilsen_real.nii');
coilsen_imag = niftiread('coilsen_imag.nii');
coilsen = complex(coilsen_real, coilsen_imag);
clear coilsen_real coilsen_imag
coilsen = single(coilsen);
% coilsen = coilsen/max(abs(coilsen(:)));


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
clearvars -except Gx Gy Gz fid Radius matrixsize coilsen
%%
%clear idx1_8 idx2_8 idx3_8 idx4_8 idx5_8
%clear idx1_10 idx2_10 idx3_10 idx4_10 idx5_10
%clear idx1_15 idx2_15 idx3_15 idx4_15 idx5_15
%clear idx1_20 idx2_20 idx3_20 idx4_20 idx5_20
load('idx1_8.mat');load('idx2_8.mat');load('idx3_8.mat');load('idx4_8.mat');load('idx5_8.mat')
index{1} = idx1_8;index{2} = idx2_8;index{3} = idx3_8;index{4} = idx4_8;index{5} = idx5_8;
% % load('idx1_10.mat');load('idx2_10.mat');load('idx3_10.mat');load('idx4_10.mat');load('idx5_10.mat')
% index{1} = idx1_10;index{2} = idx2_10;index{3} = idx3_10;index{4} = idx4_10;index{5} = idx5_10;
% load('idx1_15.mat');load('idx2_15.mat');load('idx3_15.mat');load('idx4_15.mat');load('idx5_15.mat')
% index{1} = idx1_15;index{2} = idx2_15;index{3} = idx3_15;index{4} = idx4_15;index{5} = idx5_15;
% load('idx1_20.mat');load('idx2_20.mat');load('idx3_20.mat');load('idx4_20.mat');load('idx5_20.mat')
% index{1} = idx1_20;index{2} = idx2_20;index{3} = idx3_20;index{4} = idx4_20;index{5} = idx5_20;
%%
nsamps = size(fid,1);
ncoils = size(fid,2);
nviews = size(fid,3);
% nframes = size(index,2);
nframes=1;
for i = 1:nframes
    nsampviews{i}= size(index{i},1);
end
%% filter
% %ram-lak filter for 3d k-space
w = (Radius).^2;
% %shepp-logan filter for 3d k-space
% w = (abs(Radius).*sinc(Radius)).^2;
% %low-pass pass filter for 3d k-space
% w = (abs(Radius).*cos(Radius*pi)).^2;
% % generalized hamming filter for 3d k-space
% w = ( abs(Radius).*( 0.54 + 0.46*cos(Radius*pi*2) ) ).^2;
%% angle(trajectory)
k_angle = zeros(size(fid,1),3,size(fid,3));
for i = 1:size(fid,3)
    k_angle(:,1,i) = Gx(i)*Radius;
    k_angle(:,2,i) = Gy(i)*Radius;
    k_angle(:,3,i) = Gz(i)*Radius;
end
%% data phase별로 나누기
for i = 1:nframes
    kdatau{i} = fid(:,:,index{i});
    ku{i} = k_angle(:,:,index{i});
    wu{i} = repmat(sqrt(w),[1 ncoils size(index{i},3)]);
end

for i = 1:nframes
    kdata = kdatau{i};
    kdata = kdata .* wu{i};
    kdatau{i} = kdata;
end
disp('finish making 4 phase')
disp('setup start')
%%
for nf = 1:nframes
    k = ku{nf};
    k1 = k(:,1,:);
    k2 = k(:,2,:);
    k3 = k(:,3,:);
    ktraj = [k1(:),k2(:),k3(:)];
    [kerneldistance{nf}, xyz_index{nf}, index_smth2{nf}, mask, win_3d] = setup3d(ktraj,matrixsize);
end

[IMGF] = FilterForwardGridding(mc_kdata{1},wu{1}, coilsen, kerneldistance{1}, xyz_index{1}, matrixsize, index_smth2{1}, win_3d);
figure(1); imagesc(squeeze(abs(IMGF(221:660,440,221:660))));colormap gray

% [nufft_recon] = FilterMFGridding(mc_kdata,wu, coilsen, kerneldistance, xyz_index, matrixsize, index_smth2, win_3d);
% 
% TolGrad = 1e-4;
% MaxIter = 100;
% alpha = 0.01; beta = 0.6; t0=1;
% %% initializtion
% lambda = 10*max(nufft_recon(:));
% g = gradient(nufft_recon,wu, lambda, kdatau,coilsen, nsamps, nsampviews, kerneldistance, xyz_index, index_smth2, win_3d);
% iter=0; m=nufft_recon; delta_m=-g;
% 
% %% Iterations
% while(sqrt(g(:)'*g(:)) >TolGrad && iter<8)
%     gamma_denom = g(:)'*g(:);
%     
%     t=t0;
%     f0 = FilterObjective(m,wu, lambda, kdatau,coilsen, nsamps, nsampviews, kerneldistance, xyz_index, index_smth2, win_3d);
%     f1 = FilterObjective(m+t.*delta_m,wu, lambda, kdatau,coilsen, nsamps, nsampviews, kerneldistance, xyz_index, index_smth2, win_3d);
%     i=0;
%     %backtracking line-search
%     tic
%     while(f1>f0-alpha*t*abs(g(:)'*delta_m(:)))&&(i<150)
%         t=beta*t;
%         f1 = FilterObjective(m+t.*delta_m,wu, lambda, kdatau,coilsen, nsamps, nsampviews, kerneldistance, xyz_index, index_smth2, win_3d);
%         i = i+1;
%     end
%     fprintf('line-search done\n');
%     eval(['t_linsearch',num2str(i),'= toc;'])
%     if i>2, t0=t0*beta; end
%     if i>5, t0=t0*beta*beta; end
%     if i>8, t0=t0*beta*beta; end
%     if i<2, t0=t0/beta; end
%     
%     m = m+t.*delta_m;
%     g = gradient(m,wu, lambda, kdatau,coilsen, nsamps, nsampviews, kerneldistance, xyz_index, index_smth2, win_3d);
%     gamma_num = g(:)'*g(:);
%     gamma = gamma_num/gamma_denom;
%     
%     delta_m = -g +gamma*delta_m;
%     fprintf('number of iterations: %d \n',iter+1);
%     iter = iter+1;
% end