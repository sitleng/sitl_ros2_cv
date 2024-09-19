function R = Rodrigues(rvec)
th = norm(rvec);
w = rvec/th;
w_hat = vec2hat(w);
R = eye(3)+w_hat*sin(th)+(w_hat)*(w_hat)*(1-cos(th));
end