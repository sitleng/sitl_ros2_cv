function out = gen_ecm_eqn(z1,z2,z3,z4,k,m,ths)
exp_z1 = twist2transmtx(z1,ths(1),k(1),m(1));
exp_z2 = twist2transmtx(z2,ths(2),k(2),m(2));
exp_z3 = twist2transmtx(z3,ths(3),k(3),m(3));
exp_z4 = twist2transmtx(z4,ths(4),k(4),m(4));
% function out = gen_ecm_eqn(z1,z2,z3,z4,m,ths)
% exp_z1 = twist2transmtx(z1,ths(1),m(1));
% exp_z2 = twist2transmtx(z2,ths(2),m(2));
% exp_z3 = twist2transmtx(z3,ths(3),m(3));
% exp_z4 = twist2transmtx(z4,ths(4),m(4));
% function out = gen_ecm_eqn(z1,z2,z3,z4,ths)
% exp_z1 = twist2transmtx(z1,ths(1));
% exp_z2 = twist2transmtx(z2,ths(2));
% exp_z3 = twist2transmtx(z3,ths(3));
% exp_z4 = twist2transmtx(z4,ths(4));
out = exp_z1*exp_z2*exp_z3*exp_z4;
end