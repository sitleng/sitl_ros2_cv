function out = gen_psm_eqn(z,k,m,ths)
exp_z1 = twist2transmtx(z(1:6)  ,ths(1),k(1),m(1));
exp_z2 = twist2transmtx(z(7:12) ,ths(2),k(2),m(2));
exp_z3 = twist2transmtx(z(13:18),ths(3),k(3),m(3));
exp_z4 = twist2transmtx(z(19:24),ths(4),k(4),m(4));
exp_z5 = twist2transmtx(z(25:30),ths(5),k(5),m(5));
exp_z6 = twist2transmtx(z(31:36),ths(6),k(6),m(6));
out = exp_z1*exp_z2*exp_z3*exp_z4*exp_z5*exp_z6;
end