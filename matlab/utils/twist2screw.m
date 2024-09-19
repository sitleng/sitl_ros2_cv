function [q,u,h,M] = twist2screw(v,w,th)
% S = (l=q+lambda*u, h, M)
q = cross(w,v);
h = w'*v;
u = w;
M = th;
end

