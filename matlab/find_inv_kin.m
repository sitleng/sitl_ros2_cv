%% Load dataset
clear;clc;

addpath(pathdef);

load("/home/leo/aruco_data/psm1_calib_results_final_new_v2.mat");

%%
syms ths [1 6];

z = psm1_x(1:36);
k = psm1_x(37:42);
m = psm1_x(43:48);

exp_z1 = twist2transmtx(z(1:6)  ,ths(1),k(1),m(1));
exp_z2 = twist2transmtx(z(7:12) ,ths(2),k(2),m(2));
exp_z3 = twist2transmtx(z(13:18),ths(3),k(3),m(3));
exp_z4 = twist2transmtx(z(19:24),ths(4),k(4),m(4));
exp_z5 = twist2transmtx(z(25:30),ths(5),k(5),m(5));
exp_z6 = twist2transmtx(z(31:36),ths(6),k(6),m(6));

f = gen_psm_eqn(z, k, m, ths);

%%

finv = finverse(f, ths);