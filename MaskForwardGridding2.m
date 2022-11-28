function  [IMGF] = MaskForwardGridding2(mc_kdata_gpu, coilsen, kerneldistance_gpu, xyz_index_gpu, matrixsize, index_smth2, win_3d, mask_gpu)

ncoils = 30;
kspace_data_coil = zeros(matrixsize,matrixsize,matrixsize,ncoils,'single');

kdata = repmat(mc_kdata_gpu,[64 1]);
final_value = kerneldistance_gpu.*kdata(index_smth2,:);

for i = 1:ncoils
kspace_data_coil(:,:,:,i) = accumarray(xyz_index_gpu,final_value(:,i),[matrixsize matrixsize matrixsize])./mask_gpu;
end

IMGCoil = kspace_data_coil;
IMGCoil = fftshift(fft(fftshift(IMGCoil,1),[],1),1);
IMGCoil = fftshift(fft(fftshift(IMGCoil,2),[],2),2);
IMGCoil = fftshift(fft(fftshift(IMGCoil,3),[],3),3);

IMGF = sum(coilsen.*IMGCoil,4);

IMGF = gather(IMGF./win_3d./sqrt(matrixsize*matrixsize*matrixsize));

end
