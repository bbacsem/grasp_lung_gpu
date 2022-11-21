function mf_kdata = MCMFInverseGridding3D(mf_img,w, coilsen, nsamps, nviews, kerneldistance, x_index, y_index, z_index, index_smth2, win_3d)
% image to kspace
nframes = size(mf_img,4);
ncoils = size(coilsen,4); 
mf_kdata = cell(1,nframes);

% %filter
% for Nf = 1:nframes
%     mf_kdata{Nf} = MCInverseGridding3D(mf_img(:,:,:,Nf),coilsen, nsamps, nviews{Nf}, kerneldistance{Nf}, ...
%         x_index{Nf}, y_index{Nf}, z_index{Nf}, index_smth2{Nf}, win_3d).*sqrt(repmat(w,[1 ncoils nviews]));
% end
for Nf = 1:nframes
    mf_kdata{Nf} = MCInverseGridding3D(mf_img(:,:,:,Nf),coilsen, nsamps, nviews{Nf}, kerneldistance{Nf}, ...
        x_index{Nf}, y_index{Nf}, z_index{Nf}, index_smth2{Nf}, win_3d);
end
end