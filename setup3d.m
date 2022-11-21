function [kerneldistance, x_index,y_index,z_index, index_smth2, win_3d] = setup3d(ktraj,matrixsize)
% trajectory setup
ktraj_grid = ktraj*(matrixsize-3)+(matrixsize+1)/2;

ktraj_x = ktraj_grid(:,1);
ktraj_y = ktraj_grid(:,2);
ktraj_z = ktraj_grid(:,3);

fx = floor(ktraj_x);  cx = ceil(ktraj_x); 
x_index = repmat([fx-1 fx cx cx+1],[1 16]);
fy = floor(ktraj_y); cy = ceil(ktraj_y);
y_index = horzcat(repmat(fy-1,[1 16]),repmat(fy,[1 16]),repmat(cy,[1 16]),repmat(cy+1,[1 16]));
fz = floor(ktraj_z); cz = ceil(ktraj_z);
z_index = repmat(horzcat(repmat(fz-1,[1 4]),repmat(fz,[1 4]),repmat(cz,[1 4]),repmat(cz+1,[1 4])),[1 4]);

dist_x = abs(x_index-ktraj_x); dist_y = abs(y_index-ktraj_y); dist_z = abs(z_index-ktraj_z);
distsq_x = dist_x.^2; distsq_y = dist_y.^2; distsq_z = dist_z.^2;
distsq = distsq_x + distsq_y + distsq_z;
dist = sqrt(distsq);

index_smth2 = find(dist <= 2);

x_index = single(x_index(index_smth2)); y_index = single(y_index(index_smth2)); z_index = single(z_index(index_smth2));
dist_x = dist_x(index_smth2); dist_y = dist_y(index_smth2); dist_z = dist_z(index_smth2);

beta = 5.7567;
window = kaiser(40003,beta);
lookuptable = window(20002:40003);
dist_table_x = dist_x*10000+1; dist_table_y = dist_y*10000+1; dist_table_z = dist_z*10000+1;
kerneldistance_x = (dist_table_x-floor(dist_table_x)).*lookuptable(ceil(dist_table_x))+(1-dist_table_x+floor(dist_table_x)).*lookuptable(floor(dist_table_x));
kerneldistance_y = (dist_table_y-floor(dist_table_y)).*lookuptable(ceil(dist_table_y))+(1-dist_table_y+floor(dist_table_y)).*lookuptable(floor(dist_table_y));
kerneldistance_z = (dist_table_z-floor(dist_table_z)).*lookuptable(ceil(dist_table_z))+(1-dist_table_z+floor(dist_table_z)).*lookuptable(floor(dist_table_z));
kerneldistance = kerneldistance_x.*kerneldistance_y.*kerneldistance_z;
% dist = dist(index_smth2);
% dist_table = dist*10000+1;
% kerneldistance = (dist_table-floor(dist_table)).*lookuptable(ceil(dist_table))+(1-dist_table+floor(dist_table)).*lookuptable(floor(dist_table));

%deapodization window
f = linspace(-pi,pi,matrixsize);
z = sqrt((4*f/2).^2-beta^2)+eps;
wf = abs( 4./besseli(0,beta).*sin(z)./z );
[W1,W2,W3] = meshgrid(wf,wf,wf);
win_3d = W1.*W2.*W3;
win_3d = max(win_3d,1);

xyz_index = matrixsize*matrixsize*(double(z_index)-1) + matrixsize*(double(y_index)-1) + double(x_index);

end