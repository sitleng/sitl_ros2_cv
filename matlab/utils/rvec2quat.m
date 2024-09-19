function quat = rvec2quat(rvec)
theta = norm(rvec);
if theta < 1e-9
    w = rvec;
else
    w = rvec/theta;
end
quat = zeros(4,1);
quat(1:3) = w*sin(theta/2);
quat(4) = cos(theta/2);
end

