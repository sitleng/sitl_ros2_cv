function quat = clean_quat(quat, eps)
% flipped_inds = find(abs(diff(quat(:,2))) > eps);
% out_quat = quat;
% for i = 1:length(flipped_inds)/2
%     out_quat(flipped_inds(2*i-1)+1:flipped_inds(2*i),:) = ...
%         -quat(flipped_inds(2*i-1)+1:flipped_inds(2*i),:);
% end

for i = 1:length(quat)-1
    [max_val, max_idx] = max(quat(i,:));
   if abs(quat(i+1,max_idx)-max_val) > eps
       quat(i+1,:) = -quat(i+1,:);
   end
end
end
