function R = quat2R(quat)
th = acos(quat(1))*2;
w = (quat(2:4)/sqrt(1-quat(1)^2))';
R = Rodrigues(w*th);
end

