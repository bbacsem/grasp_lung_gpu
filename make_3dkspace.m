function kspace = make_3dkspace(kvalue, index,matrixsize, ncoils )
kspace = complex(zeros(matrixsize*matrixsize*matrixsize,ncoils));
for i = 1:size(kvalue,1)
    kspace(index(i),:) = kspace(index(i),:)+ kvalue(i,:);
end
end