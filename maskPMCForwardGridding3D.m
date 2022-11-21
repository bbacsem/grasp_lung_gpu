function  [IMGF] = maskPMCForwardGridding3D(mc_kdata, coilsen, kerneldistance, xyz_index, matrixsize, index_smth2, win_3d)
gpuii = gpuArray(xyz_index);
ncoils = size(mc_kdata,2);
kspace_data_coil = zeros(matrixsize*matrixsize*matrixsize,ncoils,'single');
mask = zeros(matrixsize,matrixsize,matrixsize,'single');
uii = gather(unique(gpuii));
gpukd = gpuArray(kerneldistance);
added_kd =  groupsummary(gpukd,gpuii,'sum');
mask(uii) = added_kd;
idx_mask = find(mask ~= 0);

kdata = repmat(reshape(permute(mc_kdata,[1 3 2]),[],ncoils),[64 1]);
final_value = repmat(kerneldistance,[1 ncoils]).*kdata(index_smth2,:);

kk = zeros(size(uii,1),ncoils,'single');
for i = 1:ncoils
    gpufinalvalue = gpuArray(final_value(:,i));
    kk(:,i) =  groupsummary(gpufinalvalue,gpuii,'sum');
end
kspace_data_coil(uii,:) = kk;
kspace_data_coil(idx_mask,:) = kspace_data_coil(idx_mask,:)./repmat(mask(idx_mask)+eps*10, [1 ncoils]);
kspace_data_coil = reshape(kspace_data_coil, [matrixsize matrixsize matrixsize ncoils]);

clearvars -except coilsen IMGCoil win_3d matrixsize kspace_data_coil
IMGCoil = kspace_data_coil;
IMGCoil = fftshift(fft(fftshift(IMGCoil,1),[],1),1);
IMGCoil = fftshift(fft(fftshift(IMGCoil,2),[],2),2);
IMGCoil = fftshift(fft(fftshift(IMGCoil,3),[],3),3);

% IMGF = sum(conj(coilsen).*IMGCoil,4);
IMGF = sum(abs(IMGCoil).^2,4);
IMGF = IMGF./win_3d./sqrt(matrixsize*matrixsize*matrixsize);
end