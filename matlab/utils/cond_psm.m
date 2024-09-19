function [c,ceq] = cond_psm(x)
c=[];
ceq(1) = norm(x(4:6))-1;
ceq(2) = norm(x(10:12))-1;
ceq(3) = norm(x(13:15))-1;
ceq(4) = norm(x(16:18));
ceq(5) = norm(x(22:24))-1;
ceq(6) = norm(x(28:30))-1;
ceq(7) = norm(x(34:36))-1;
end