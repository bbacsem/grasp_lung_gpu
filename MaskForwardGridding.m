function  [IMGF] = MaskForwardGridding(mc_kdata, coilsen, kerneldistance, xyz_index, matrixsize, index_smth2, win_3d, mask)

xyz_index_gpu = gpuArray(double(xyz_index));
kerneldistance_gpu = gpuArray(kerneldistance);
IMGF_gpu = zeros(matrixsize,matrixsize,matrixsize,'single');
ncoils = size(mc_kdata,2);

mask_idx_gpu = gpuArray(mask.val_idx);
mask_val_gpu = gpuArray(mask.val);

for i = 1:ncoils
kdata = gpuArray(repmat(reshape(squeeze(mc_kdata(:,i,:)),[],1),[64 1]));
kdata = kdata(index_smth2);
final_value = kerneldistance_gpu.*kdata;

kspace_gpu = accumarray(xyz_index_gpu,final_value,[matrixsize matrixsize matrixsize]);
kspace_gpu(mask_idx_gpu) = kspace_gpu(mask_idx_gpu)./mask_val_gpu;

IMG = fftshift(fft(fftshift(kspace_gpu,1),[],1),1);
IMG = fftshift(fft(fftshift(IMG,2),[],2),2);
IMG = fftshift(fft(fftshift(IMG,3),[],3),3);

IMGF_gpu = IMGF_gpu + IMG.*gpuArray(coilsen(:,:,:,i));
end

IMGF = gather(IMGF_gpu./win_3d./sqrt(matrixsize*matrixsize*matrixsize));

end
