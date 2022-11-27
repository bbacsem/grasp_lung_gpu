function obj = FilterObjective(input_img,wu, lambda, y,coilsen, nsamps, nviews, kerneldistance, xyz_index, index_smth2, win_3d)
n = size(input_img,4);
input = reshape(input_img,[],n);
input = transpose(input);
tv_temp = zeros(n);
L2Obj =0;
for i = 1:n
    for j = 1:n
        if i==j
            tv_temp(i,j) = -1;
        elseif j == i+1
            tv_temp(i,j) = 1;
        else
            tv_temp(i,j) = 0;
        end
    end
end
tv_temp(n,n) = 0;
sparse_mat = tv_temp*input;
L1Obj = lambda.*sum(abs(sparse_mat(:)));

kspace = FilterMFInverseGridding(input_img,wu,coilsen, nsamps, nviews, kerneldistance, xyz_index, index_smth2, win_3d);
for i = 1:n
    kk = kspace{i};
    yy = y{i};
    L2Obj = L2Obj + (kk(:)-yy(:))'*(kk(:)-yy(:));
end
obj = L1Obj + L2Obj;
end
%test
