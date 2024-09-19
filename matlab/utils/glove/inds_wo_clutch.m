function res = inds_wo_clutch(inds, box_start, box_end)
clutch_inds = zeros(size(inds));
for i = 1:numel(box_start)
    clutch_inds = clutch_inds | (inds >= box_start(i) & inds <= box_end(i));
end
res = ~clutch_inds;
end
