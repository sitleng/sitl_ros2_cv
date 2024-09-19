function out = hat2vec(in)
[r, c] = size(in);
if r~=3 || c~=3
    error("Incorrect matrix size, it should be 3x3")
end
if (in(3,2)~=-in(2,3) || in(1,3)~=-in(3,1) || in(2,1)~=-in(1,2))
    error("The matrix is not skew symmetric")
end
out = [in(3,2);in(1,3);in(2,1)];
end

