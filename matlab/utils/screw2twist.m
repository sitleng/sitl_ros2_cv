function [v,w,th] = screw2twist(q,u,h,M)
th = M;
w = u/norm(u);
v = -cross(w,q) + h*w;
end

