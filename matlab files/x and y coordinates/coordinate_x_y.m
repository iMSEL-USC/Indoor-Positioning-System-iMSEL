clc;clear;
syms x y
% distance from this point to point A
dist_A = 9.5;
% distance from this point to point B
dist_B = 129.8;
% coordinates of point A
benchmark_point_A = [243.84,243.84];
% coordinates of point B
benchmark_point_B = [365.76,243.84];


S = vpasolve([(x - benchmark_point_A(1))^2 + (y - benchmark_point_A(2))^2 == dist_A^2, (x - benchmark_point_B(1))^2 + (y - benchmark_point_B(2))^2 == dist_B^2], [x,y]);
disp("x_coordinate: ")
disp(S.x)
disp("y_coordinate: ")
disp(S.y)