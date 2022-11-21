function  [IMGF] = MCMFGridding3D(mcmf_kdata, w,coilsen, kerneldistance, x_index, y_index, z_index, matrixsize, index_smth2, win_3d)
%ksapce to image
nframes = size(mcmf_kdata,2);
% ncoils = size(coilsen,4);
IMGF = zeros(matrixsize, matrixsize, matrixsize, nframes);

%%filter
% for Nf = 1:nframes
%     mc_kdata = mcmf_kdata{Nf};
%     nviews = size(mc_kdata,3);
%     IMGF(:,:,:,Nf) = MCForwardGridding3D(mc_kdata.*sqrt(repmat(w,[1 ncoils nviews])), coilsen, kerneldistance{Nf}, x_index{Nf}, y_index{Nf}, z_index{Nf}, matrixsize, index_smth2{Nf}, win_3d);
% end

%mask
for Nf = 1:nframes
    mc_kdata = mcmf_kdata{Nf};
    IMGF(:,:,:,Nf) = MCForwardGridding3Daccum(mc_kdata, coilsen, kerneldistance{Nf}, x_index{Nf}, y_index{Nf}, z_index{Nf}, matrixsize, index_smth2{Nf}, win_3d);
end

end
