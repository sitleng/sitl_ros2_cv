function g = twist2transmtx(xi,th,k,m)
v = xi(1:3);
w = xi(4:6);
eps = 1e-6;
th = k*th+m;
w_norm = norm(w);
if w_norm <= eps
    R = eye(3);
    v = v/norm(v);
    p = v*th;
else
    w = w/w_norm;
    v = v/w_norm;
    % th = th*w_norm;
    R = Rodrigues(w*th);
    p = (eye(3)-R)*cross(w,v)+w*w'*v*th;
end
% function g = twist2transmtx(xi,th,m)
% v = xi(1:3);
% w = xi(4:6);
% eps = 1e-6;
% th = th+m;
% w_norm = norm(w);
% if w_norm <= eps
%     R = eye(3);
%     v = v/norm(v);
%     p = v*th;
% else
%     w = w/w_norm;
%     v = v/w_norm;
%     % th = th*w_norm;
%     R = Rodrigues(w*th);
%     p = (eye(3)-R)*cross(w,v)+w*w'*v*th;
% end
% function g = twist2transmtx(xi,th)
% v = xi(1:3);
% w = xi(4:6);
% eps = 1e-6;
% w_norm = norm(w);
% if w_norm <= eps
%     R = eye(3);
%     p = v*th;
% else
%     w = w/w_norm;
%     v = v/w_norm;
%     th = th*w_norm;
%     R = Rodrigues(w*th);
%     p = (eye(3)-R)*cross(w,v)+w*w'*v*th;
% end
g = [R p;0 0 0 1];
end