function  [IMGF] = MaskMFGridding(mcmf_kdata, coilsen, kerneldistance, x_index, y_index, z_index, matrixsize, index_smth2, win_3d)
%ksapce to image
nframes = size(mcmf_kdata,2);
IMGF = zeros(matrixsize, matrixsize, matrixsize, nframes,'single');

for Nf = 1:nframes
    mc_kdata = mcmf_kdata{Nf};
    IMGF(:,:,:,Nf) = MCForwardGridding3Dgpu(mc_kdata, coilsen, kerneldistance{Nf}, x_index{Nf}, y_index{Nf}, z_index{Nf}, matrixsize, index_smth2{Nf}, win_3d);
end

end
