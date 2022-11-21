function  [mc_kdata] = MCInverseGridding3Dgpu(img,coilsen, nsamps, nviews, kerneldistance, x_index, y_index, z_index, index_smth2, win_3d)
%multicoil inverse gridding
%image to kspace
ncoils = size(coilsen,4);
matrixsize = size(img,1);
mc_kdata = zeros(nsamps,ncoils,nviews);
index = gpuArray(matrixsize*matrixsize*(z_index-1) + matrixsize*(y_index-1) + x_index);

img_gpu = gpuArray(img);
kerneldistance_gpu = gpuArray(kerneldistance);
index_smth2_gpu = gpuArray(index_smth2);

tic
for Nc = 1:ncoils
    coilsen_gpu = gpuArray(coilsen(:,:,:,Nc));
    img1 = conj(coilsen_gpu).*img_gpu;
    % img = img./win_3d;
    kspace_data_grid = fftshift(fft(fftshift(img1,1),[],1),1);
    kspace_data_grid = fftshift(fft(fftshift(kspace_data_grid,2),[],2),2);
    kspace_data_grid = fftshift(fft(fftshift(kspace_data_grid,3),[],3),3);

    kspace_data_grid = kspace_data_grid(:);
    kspace_data_grid = kspace_data_grid(index);
    kdata_rad = kspace_data_grid.*kerneldistance_gpu;
    
    kdata = gpuArray(zeros(nsamps*nviews,64));
    kdata(index_smth2_gpu) = kdata_rad;
    kdata = reshape(kdata,[nsamps nviews 64]);
    kdata = sum(kdata,3);
    mc_kdata(:,Nc,:) = gather(kdata./sqrt(matrixsize*matrixsize*matrixsize));
end
toc
end