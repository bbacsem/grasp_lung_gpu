function  [IMGF] = FilterForwardGridding(mc_kdata,wu, coilsen, kerneldistance, xyz_index, matrixsize, index_smth2, win_3d)

ncoils = size(mc_kdata,2);
mc_kdata = mc_kdata.*wu;

xyz_index_gpu = gpuArray(double(xyz_index));
kerneldistance_gpu = gpuArray(kerneldistance);

IMGF_gpu = gpuArray(zeros(matrixsize,matrixsize,matrixsize,'single'));

for i = 1:ncoils
coil = gpuArray(coilsen(:,:,:,i));

kdata = gpuArray(repmat(reshape(squeeze(mc_kdata(:,i,:)),[],1),[64 1]));
kdata = kdata(index_smth2);
final_value = kerneldistance_gpu.*kdata;

kspace_gpu = accumarray(xyz_index_gpu,final_value,[matrixsize matrixsize matrixsize]);

IMG = fftshift(fft(fftshift(kspace_gpu,1),[],1),1);
IMG = fftshift(fft(fftshift(IMG,2),[],2),2);
IMG = fftshift(fft(fftshift(IMG,3),[],3),3);

IMGF_gpu = IMGF_gpu + IMG.*coil;

end
IMGF = gather(IMGF_gpu./sqrt(matrixsize*matrixsize*matrixsize));
end
