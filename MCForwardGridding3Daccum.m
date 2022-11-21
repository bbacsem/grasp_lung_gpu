function  [IMGF] = MCForwardGridding3Daccum(mc_kdata, coilsen, kerneldistance, x_index, y_index, z_index, matrixsize, index_smth2, win_3d)

ncoils = size(mc_kdata,2);
kspace_data_coil = zeros(matrixsize,matrixsize,matrixsize,ncoils,'single');

xyz_index = cat(2,x_index,y_index,z_index);
mask = accumarray(double(xyz_index),kerneldistance,[matrixsize matrixsize matrixsize]);
idx_mask = find(mask ~= 0);
mask_num = mask(idx_mask)+10*eps;

mc_kdata = reshape(permute(mc_kdata,[1 3 2]),[],ncoils);
mc_kdata = repmat(mc_kdata,[64 1]);
kdata = mc_kdata(index_smth2,:);
final_value = repmat(kerneldistance,[1 ncoils]).*kdata;

for i = 1:ncoils
kspace = accumarray(double(xyz_index),double(final_value(:,i)),[matrixsize matrixsize matrixsize]);
kspace(idx_mask) = kspace(idx_mask)./mask_num;
kspace_data_coil(:,:,:,i) = kspace;
end

clearvars -except coilsen IMGCoil win_3d matrixsize kspace_data_coil
IMGCoil = kspace_data_coil;
IMGCoil = fftshift(fft(fftshift(IMGCoil,1),[],1),1);
IMGCoil = fftshift(fft(fftshift(IMGCoil,2),[],2),2);
IMGCoil = fftshift(fft(fftshift(IMGCoil,3),[],3),3);

IMGF = sum(coilsen.*IMGCoil,4);
IMGF = IMGF./win_3d./sqrt(matrixsize*matrixsize*matrixsize);
end
