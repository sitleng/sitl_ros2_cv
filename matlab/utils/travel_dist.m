function total_distance = travel_dist(P)
    if size(P,2)~=3 && size(P,1)==3
        P = P';
    end
    
    % Calculate the differences between consecutive points
    differences = diff(P);
    
    % Calculate the Euclidean distance between each consecutive pair of points
    distances = sqrt(sum(differences.^2, 2));
    
    % Sum all distances to get the total traveled distance
    total_distance = sum(distances);
end

