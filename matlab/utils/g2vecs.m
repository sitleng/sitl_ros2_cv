function [rvec,tvec] = g2vecs(g)
tvec = g(1:3,4);
rvec = RodriguesInv(g(1:3,1:3));
end

