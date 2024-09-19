function [trans,quat] = g2transquat(g)
trans = g(1:3,4);
rvec = RodriguesInv(g(1:3,1:3));
quat = rvec2quat(rvec);
end

