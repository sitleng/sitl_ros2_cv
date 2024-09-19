function [R,w,th] = exp2rot(exp_coord)
th = norm(exp_coord);
w = exp_coord/th;
R = rodrigues(w,th);
end

