function [v,w,th] = SE3_to_se3(g)
R = g(1:3,1:3);
p = g(1:3,4);
[w,th] = rot2exp(R);
v = ((eye(3)-R)*vec2hat(w)+w*w'*th)\p;
end