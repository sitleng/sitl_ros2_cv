function [c,ceq] = cond_ecm(x)
c=[];
% c(1) = -x(33)+0.01;
% c(2) = -x(34)+0.01;

% ceq(1) = norm(x(13:15))-1;

ceq(1) = norm(x(4:6))-1;
ceq(2) = norm(x(10:12))-1;
ceq(3) = norm(x(13:15))-1;
ceq(4) = norm(x(16:18));
ceq(5) = norm(x(22:24))-1;
% ceq(6) = x(33)+x(34)-2;
end