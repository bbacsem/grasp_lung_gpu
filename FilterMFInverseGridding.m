function mf_kdata = FilterMFInverseGridding(mf_img,wu, coilsen, nsamps, nviews, kerneldistance, xyz_index, index_smth2, win_3d)
% image to kspace
nframes = size(mf_img,4);
mf_kdata = cell(1,nframes);

for Nf = 1:nframes
    tic
    mf_kdata{Nf} = FilterInverseGridding(mf_img(:,:,:,Nf), wu{Nf},coilsen, nsamps, nviews{Nf}, kerneldistance{Nf}, xyz_index{Nf}, index_smth2{Nf}, win_3d);
    toc
end

end