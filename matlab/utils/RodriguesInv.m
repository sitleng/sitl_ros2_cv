function rvec = RodriguesInv(R)
th = acos((trace(R)-1)/2);
if sin(th) < 1e-6
    rvec = zeros(3,1);
else
    w = (1/(2*sin(th)))*[R(3,2) - R(2,3);
                        R(1,3) - R(3,1);
                        R(2,1) - R(1,2)];
    rvec = w*th;
end
end