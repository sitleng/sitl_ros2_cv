function out = exp2hat(in)
v = in(1:3);
w = in(4:6);
out = [vec2hat(w) v;0 0 0 0];
end

