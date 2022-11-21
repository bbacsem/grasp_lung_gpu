function grad = gradient(input_img, w, lambda, y,coilsen, nsamps, nviews, kerneldistance, x_index, y_index, z_index, index_smth2, win_3d)
% L2grad
kspace = MCMFInverseGridding3D(input_img, w, coilsen, nsamps, nviews, kerneldistance, x_index, y_index, z_index, index_smth2, win_3d);
diff_kspace = cell(1,5);
for i = 1:size(input_img,4)
diff_kspace{i} = kspace{i}-y{i};
end
matrixsize = size(input_img,1);
recon = MCMFGridding3D(diff_kspace, w,coilsen, kerneldistance, x_index, y_index, z_index, matrixsize, index_smth2, win_3d);
L2grad = 2.*recon;

%L1grad
img_size = size(input_img,1);
nframes = size(input_img,4);
input = reshape(input_img,[],nframes);
input = transpose(input);
% TV_temp
tv_temp = zeros(nframes);
for i = 1:nframes
    for j = 1:nframes
        if i==j
            tv_temp(i,j) = -1;
        elseif j == i+1
            tv_temp(i,j) = 1;
        else
            tv_temp(i,j) = 0;
        end
    end
end
tv_temp(nframes,nframes) = 0;
smoothing_param = 1e-15;
sparse_mat = tv_temp*input;
L1grad = lambda*tv_temp'*(sparse_mat.*(conj(sparse_mat).*sparse_mat+smoothing_param).^(-0.5));
L1grad = transpose(L1grad);
L1grad = reshape(L1grad, img_size,img_size,img_size,[]);

grad = L1grad +L2grad;


end
