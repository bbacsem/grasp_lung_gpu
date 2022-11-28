function  [IMGF] = MaskMFGridding(mcmf_kdata, coilsen, kerneldistance, xyz_index, matrixsize, index_smth2, win_3d, mask)
%ksapce to image
nframes = size(mcmf_kdata,2);
IMGF = zeros(matrixsize, matrixsize, matrixsize, nframes,'single');

for Nf = 1:nframes
    mc_kdata = mcmf_kdata{Nf}; 
    IMGF(:,:,:,Nf) = MCForwardGridding3Dgpu(mc_kdata, coilsen, kerneldistance{Nf}, xyz_index{Nf}, matrixsize, index_smth2{Nf}, win_3d, mask{Nf});
end

end