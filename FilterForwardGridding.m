function  [IMGF] = FilterForwardGridding(mc_kdata, coilsen, kerneldistance, xyz_index, matrixsize, index_smth2, win_3d)

xyz_index_gpu = gpuArray(double(xyz_index));
kerneldistance_gpu = gpuArray(kerneldistance);

ncoils = size(mc_kdata,2);
IMGCoil = zeros(matrixsize,matrixsize,matrixsize,ncoils,'single');

% xyz_index = cat(2,x_index,y_index,z_index);
for i = 1:ncoils

kdata = gpuArray(repmat(reshape(squeeze(mc_kdata(:,i,:)),[],1),[64 1]));
kdata = kdata(index_smth2);
final_value = kerneldistance_gpu.*kdata;

kspace_gpu = accumarray(xyz_index_gpu,final_value,[matrixsize matrixsize matrixsize]);

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
