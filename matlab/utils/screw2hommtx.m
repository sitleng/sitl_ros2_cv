function [w,R,p,g] = screw2hommtx(q,u,h,M)
w = norm(u);
if abs(w-1) > 1e-3
    new_u = u/w;
else
	new_u = u;
end
R = rodrigues(new_u,M);
p = (eye(3)-R)*q + h*new_u*M;
g = [R p;0 0 0 1];
end

