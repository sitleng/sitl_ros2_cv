function dist = Rdist(R1, R2)
sz1 = size(R1);
sz2 = size(R2);
if ~all(sz1 == sz2) || (length(sz1) < 2) || (length(sz1) > 3)  
    dist = NaN;
else
    N = sz1(end);
    dist = nan([1,N]);
    for i = 1:N
        dist(i) = acos((trace(R1(:,:,i)*R2(:,:,i)')-1)/2);
    end
end
