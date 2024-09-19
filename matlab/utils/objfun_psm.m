function f = objfun_psm(x,N,trans,joint_ths)
errors = zeros(N,1);
% z1 = x(1:6);
% z2 = x(7:12);
% z3 = x(13:18);
% z3(4:6) = zeros(3,1);
% z4 = x(19:24);
% z5 = x(25:30);
% z6 = x(31:36);

z = x(1:36);
z(16:18) = zeros(3,1);
k  = x(37:42);
m  = x(43:48);
w  = [0.7,0.3];
for i=1:N
    tran = trans(:,:,i);
    eqn  = gen_psm_eqn(z,k,m,joint_ths(i,1:end));
    % p_eqn = eqn(1:3,4);
    % R_eqn = eqn(1:3,1:3);
    % p_tran = tran(1:3,4);
    % R_tran = tran(1:3,1:3);
    % Manhattan: p=1, Euclidean: p=2, Chebychev: p=inf
    % distT = norm(p_tran-p_eqn,1);
    % distR = acos((trace(R_tran*R_eqn')-1)/2);
    % errors(i) = w(1)*distT + w(2)*distR;
    errors(i) = dist(se3(tran), se3(eqn), w);
end
f = sum(errors.^2)/N;
end