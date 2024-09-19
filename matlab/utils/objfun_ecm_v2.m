function f = objfun_ecm_v2(x,xdata)

z1 = x(1:6);
z2 = x(7:12);
z3 = x(13:18);
z3(4:6) = zeros(3,1);
z4 = x(19:24);
k  = x(25:28);
m  = x(29:32);
g  = gen_ecm_eqn(z1,z2,z3,z4,k,m,xdata);
[trans, quat]  = g2transquat(g);
f = [trans;quat];
end