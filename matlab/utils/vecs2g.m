function g = vecs2g(rvec,tvec)
R = Rodrigues(rvec);
g = [R tvec;0 0 0 1];
end

