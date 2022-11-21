function  [IMGF] = MCForwardGridding3D(mc_kdata, coilsen, kerneldistance, x_index, y_index, z_index, matrixsize, index_smth2, win_3d)

ncoils = size(mc_kdata,2);
IMGCoil = zeros(matrixsize,matrixsize,matrixsize,ncoils);
xyz_index = cat(2,x_index,y_index,z_index);
for i = 1:ncoils
    kdata = mc_kdata(:,i,:);
    kdata = kdata(:);
    kdata = repmat(kdata,[1 64]);
    kdata = kdata(index_smth2);
    final_value = kerneldistance.*kdata;
    %mask
    [kspace, mask] = GridSetUpSingle(single(final_value'), single(kerneldistance'),uint16(x_index'),uint16(y_index'),uint16(z_index'),matrixsize);
%     kspace = accumarray(xyz_index,final_value,[matrixsize matrixsize matrixsize],@sum);
%     mask = accumarray(xyz_index,kerneldistance,[matrixsize matrixsize matrixsize],@sum);
    idx_mask = find(mask ~= 0);
    kspace(idx_mask) = kspace(idx_mask)./(mask(idx_mask)+eps*10);
    kspace_data = reshape(kspace,[matrixsize matrixsize matrixsize]);
    %filter
%     [kspace] = GridSetUpSingleOnlyfid(single(final_value'),uint16(x_index'),uint16(y_index'),uint16(z_index'),matrixsize);
%     kspace_data = reshape(kspace,[matrixsize matrixsize matrixsize]);

    IMG1 = kspace_data;
    IMG1 = fftshift(fft(fftshift(IMG1,1),[],1),1);
    IMG1 = fftshift(fft(fftshift(IMG1,2),[],2),2);
    IMG1 = fftshift(fft(fftshift(IMG1,3),[],3),3);

%     IMG = IMG1./win_3d./sqrt(matrixsize*matrixsize*matrixsize);
    IMGCoil(:,:,:,i) = IMG1;
end
clearvars -except IMGCoil
IMGF = sqrt(sum(IMGCoil.^2,4));

% IMGF = sqrt(sum(abs(IMGCoil).^2,4));
end