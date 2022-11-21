function [coilsen] = CoilSensitivityGenerator(IMGCoil)

% IMGCoil = fftshift(fft(fftshift(IMGCoil,1),[],1),1);
% IMGCoil = fftshift(fft(fftshift(IMGCoil,2),[],2),2);
% IMGCoil = fftshift(fft(fftshift(IMGCoil,3),[],3),3);
%LPF - Gaussian filter
% f = gausswin(matrixsize);
% [W1,W2,W3] = meshgrid(f,f,f);
% gauss_win = W1.*W2.*W3;
% for i = 1:ncoils
%     IMGCoil(:,:,:,i) = gauss_win.*IMGCoil(:,:,:,i);
% end
% IMGCoil = fftshift(fft(fftshift(IMGCoil,1),[],1),1);
% IMGCoil = fftshift(fft(fftshift(IMGCoil,2),[],2),2);
% IMGCoil = fftshift(fft(fftshift(IMGCoil,3),[],3),3);
coilsen = zeros(matrixsize,matrixsize,matrixsize,ncoils);
ss = sqrt(sum(abs(IMGCoil.^2),4));
for j = 1:30
    tic
    coilsen(:,:,:,j) = IMGCoil(:,:,:,j)./ss;
    toc
end
nii = make_nii(real(coilsen));
save_nii(nii,'coilsen_real.nii');
nii = make_nii(imag(coilsen));
save_nii(nii,'coilsen_imag.nii');
end



% %generate coilsensitivity
% % CoilSensitivityGenerator
% 
% csdata = kdatau{1};
% ref_img = zeros(matrixsize,matrixsize,matrixsize,ncoils);
% for i = 1:30
% %     csdata1(:,:,i) = squeeze(csdata(:,i,:));
% %     csdata2(:,i) = reshape(csdata1(:,:,i),[],1);
% %     ref_img(:,:,:,i) = ForwardGridding3D(csdata2(:,i), kerneldistance, x_index, y_index, z_index, matrixsize, index_smth2, win_3d);
%     csdata1 = squeeze(csdata(:,i,:));
%     csdata1 = reshape(csdata1,[],1);
%     ref_img(:,:,:,i) = ForwardGridding3D(csdata1, kerneldistance{1}, x_index{1}, y_index{1}, z_index{1}, matrixsize, index_smth2{1}, win_3d);
% end
% ss = sqrt(sum(abs(ref_img.^2),4));
% for j = 1:30
%     tic
%     ref_img(:,:,:,j) = ref_img(:,:,:,j)./ss;
%     toc
% end
% % coilsen = ref_img./repmat(sqrt(sum(abs(ref_img.^2),4)),[1 1 1 ncoils]);
% % 메모리 주의
% figure(30); imagesc(abs(squeeze(ref_img(:,:,440,30))));
% 
% nii = make_nii(real(ref_img));
% save_nii(nii,'coilsen_real.nii');
% 
% nii = make_nii(imag(ref_img));
% save_nii(nii,'coilsen_imag.nii');
% 
% % coilsen_real = niftiread('coilsen_real.nii');
% % coilsen_imag = niftiread('coilsen_imag.nii');
% % coilsen = complex(coilsen_real, coilsen_imag);
% % coilsen = single(coilsen);
% % coilsen = coilsen/max(abs(coilsen(:)));
% % clear coilsen_real coilsen_imag


