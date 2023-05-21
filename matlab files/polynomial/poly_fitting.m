close all; clear; clc;

%% Load Data

traindata = load('D:\OUJ\OneDrive - University of South Carolina\OUJ\paper\3\code\data\training_40.mat');
testdata = load('D:\OUJ\OneDrive - University of South Carolina\OUJ\paper\3\code\data\testing.mat');


% Train Set
Xc_black = traindata.Xc_black;
Xc_red = traindata.Xc_red;
Xc_white = traindata.Xc_white;
Xw_black = traindata.Xw_black;
Xw_red = traindata.Xw_red;
Xw_white = traindata.Xw_white;

% Test Set
Xc_black_test = testdata.Xc_black_test;
Xc_red_test = testdata.Xc_red_test;
Xc_white_test = testdata.Xc_white_test;
Xw_black_test = testdata.Xw_black_test;
Xw_red_test = testdata.Xw_red_test;
Xw_white_test = testdata.Xw_white_test;

Xc_black = [Xc_black;Xc_black_test];
Xc_red = [Xc_red;Xc_red_test];
Xc_white = [Xc_white;Xc_white_test];
Xw_black = [Xw_black;Xw_black_test];
Xw_red = [Xw_red;Xw_red_test];
Xw_white = [Xw_white;Xw_white_test];


n = size(Xc_red,1);
A_red = Xc_red';
B_red = Xw_red';
A_white = Xc_white';
B_white = Xw_white';
A_black = Xc_black';
B_black = Xw_black';
% Find Rotational Matrix and Transformation Vector (RT+poly)
% [Rmat_red, Tvec_red] = rigid_transform_3D(A_red, B_red);
% [Rmat_white, Tvec_white] = rigid_transform_3D(A_white, B_white);
% [Rmat_black, Tvec_black] = rigid_transform_3D(A_black, B_black);
% Coordinate Transformation
% B2_red = (Rmat_red*A_red) + repmat(Tvec_red, 1, n);
% B2_white = (Rmat_white*A_white) + repmat(Tvec_white, 1, n);
% B2_black = (Rmat_black*A_black) + repmat(Tvec_black, 1, n);
% 
% Xc_red = B2_red';
% Xc_white = B2_white';
% Xc_black = B2_black';
% 
% n = size(Xc_red_test,1);
% B2_red = (Rmat_red*Xc_red_test') + repmat(Tvec_red, 1, n);
% B2_white = (Rmat_white*Xc_white_test') + repmat(Tvec_white, 1, n);
% B2_black = (Rmat_black*Xc_black_test') + repmat(Tvec_black, 1, n);
% 
% Xc_red_test = B2_red';
% Xc_white_test = B2_white';
% Xc_black_test = B2_black';

p_x_red = polyfitn(Xc_red, Xw_red(:,1),'constant, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2');
vpa(polyn2sym(p_x_red),4)
p_y_red = polyfitn(Xc_red, Xw_red(:,2),'constant, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2');
vpa(polyn2sym(p_y_red),4)
p_z_red = polyfitn(Xc_red, Xw_red(:,3),'constant, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2');
vpa(polyn2sym(p_z_red),4)

p_x_white = polyfitn(Xc_white(:,1:3), Xw_white(:,1),'constant, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2');
vpa(polyn2sym(p_x_white),4)
p_y_white = polyfitn(Xc_white(:,1:3), Xw_white(:,2),'constant, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2');
vpa(polyn2sym(p_y_white),4)
p_z_white = polyfitn(Xc_white, Xw_white(:,3),'constant, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2');
vpa(polyn2sym(p_z_white),4)

p_x_black = polyfitn(Xc_black(:,1:3), Xw_black(:,1),'constant, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2');
vpa(polyn2sym(p_x_black),4)
p_y_black = polyfitn(Xc_black(:,1:3), Xw_black(:,2),'constant, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2');
vpa(polyn2sym(p_y_black),4)
p_z_black = polyfitn(Xc_black, Xw_black(:,3),'constant, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2');
vpa(polyn2sym(p_z_black),4)


XW_red_poly = polyvaln(p_x_red,Xc_red_test);
YW_red_poly = polyvaln(p_y_red,Xc_red_test);
ZW_red_poly = polyvaln(p_z_red,Xc_red_test);
difference_red = [XW_red_poly,YW_red_poly,ZW_red_poly] - Xw_red_test;
difference_red = difference_red(:,1:2);
DE_red = sqrt(sum(difference_red.^2,2));
mean_red = mean(DE_red)

XW_white_poly = polyvaln(p_x_white,Xc_white_test);
YW_white_poly = polyvaln(p_y_white,Xc_white_test);
ZW_white_poly = polyvaln(p_z_white,Xc_white_test);
difference_white = [XW_white_poly,YW_white_poly,ZW_white_poly] - Xw_white_test;
difference_white = difference_white(:,1:2);
DE_white = sqrt(sum(difference_white.^2,2));
mean_white = mean(DE_white)

XW_black_poly = polyvaln(p_x_black,Xc_black_test);
YW_black_poly = polyvaln(p_y_black,Xc_black_test);
ZW_black_poly = polyvaln(p_z_black,Xc_black_test);
difference_black = [XW_black_poly,YW_black_poly,ZW_black_poly] - Xw_black_test;
difference_black = difference_black(:,1:2);
DE_black = sqrt(sum(difference_black.^2,2));
mean_black = mean(DE_black)

min_red = min(DE_red);
min_white = min(DE_white);
min_black = min(DE_black);
max_red = max(DE_red);
max_white = max(DE_white);
max_black = max(DE_black);
std_red = std(DE_red);
std_white = std(DE_white);
std_black = std(DE_black);