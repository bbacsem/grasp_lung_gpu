function  [IMGF] = MCForwardGridding3Dgpu(mc_kdata, coilsen, kerneldistance, xyz_index, matrixsize, index_smth2, win_3d)

xyz_index_gpu = gpuArray(double(xyz_index));
kerneldistance_gpu = gpuArray(kerneldistance);

ncoils = size(mc_kdata,2);
IMGCoil = zeros(matrixsize,matrixsize,matrixsize,ncoils,'single');

% xyz_index = cat(2,x_index,y_index,z_index);
mask_gpu = accumarray(xyz_index_gpu,kerneldistance_gpu,[matrixsize matrixsize matrixsize]);
idx_mask_gpu = find(mask_gpu ~= 0);
mask_num_gpu = mask_gpu(idx_mask_gpu)+10*eps;

% mc_kdata = reshape(permute(mc_kdata,[1 3 2]),[],ncoils);
% mc_kdata = repmat(mc_kdata,[64 1]);
% kdata = mc_kdata(index_smth2,:);
% final_value = repmat(kerneldistance_gpu,[1 ncoils]).*kdata;

for i = 1:ncoils

kdata = gpuArray(repmat(reshape(squeeze(mc_kdata(:,i,:)),[],1),[64 1]));
kdata = kdata(index_smth2);
final_value = kerneldistance_gpu.*kdata;

kspace_gpu = accumarray(xyz_index_gpu,final_value,[matrixsize matrixsize matrixsize]);
kspace_gpu(idx_mask_gpu) = kspace_gpu(idx_mask_gpu)./mask_num_gpu;

IMG = fftshift(fft(fftshift(kspace_gpu,1),[],1),1);
IMG = fftshift(fft(fftshift(IMG,2),[],2),2);
IMG = fftshift(fft(fftshift(IMG,3),[],3),3);

IMGCoil(:,:,:,i) = gather(IMG);
end

clearvars -except coilsen IMGCoil win_3d matrixsize 
% IMGF = sum(coilsen.*IMGCoil,4);

IMGF = sqrt(sum(abs(IMGCoil).^2,4));
IMGF = IMGF./win_3d./sqrt(matrixsize*matrixsize*matrixsize);
end
