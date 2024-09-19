function f = objfun_ecm(x,N,trans,joint_ths)
errors = zeros(N,1);
z1 = x(1:6);
z2 = x(7:12);
z3 = x(13:18);
z3(4:6) = zeros(3,1);
z4 = x(19:24);
k  = x(25:28);
m  = x(29:32);
% m  = x(25:28);
w  = [0.7,0.3];
% w = x(33:34);
for i=1:N
    tran = trans(:,:,i);
    eqn  = gen_ecm_eqn(z1,z2,z3,z4,k,m,joint_ths(i,1:end));
    % eqn  = gen_ecm_eqn(z1,z2,z3,z4,m,joint_ths(i,1:end));
    p_eqn = eqn(1:3,4);
    R_eqn = eqn(1:3,1:3);
    p_tran = tran(1:3,4);
    R_tran = tran(1:3,1:3);
    % Manhattan: p=1, Euclidean: p=2, Chebychev: p=inf
    distT = norm(p_tran-p_eqn,1);
    distR = acos((trace(R_tran*R_eqn')-1)/2);
    errors(i) = w(1)*distT + w(2)*distR;
end
f = sum(errors.^2)/N;
end