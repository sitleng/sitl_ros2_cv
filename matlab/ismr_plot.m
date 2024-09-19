%%
clear;

%% Load dataset
clc;

addpath(pathdef);

filename = sprintf('C:\\Users\\kiwil\\Downloads\\List.csv');
pred_res = readtable(filename,"VariableNamingRule","preserve");
pred_lgbm = pred_res.("L-GBM");
pred_rf = pred_res.("Random Forest");
pred_ada = pred_res.("ADA");
ground_truth = pred_res.("GD");

clutch_start = find(diff(ground_truth)>0);
clutch_end   = find(diff(ground_truth)<0);

N = 10000;
% N = length(ground_truth);
t = 1:N;

wo_clutch = inds_wo_clutch(N, clutch_start, clutch_end);

%%

tl = tiledlayout('flow');
ax = nexttile;
hold on;

ax = plot_tile(ax, t, pred_lgbm(t), 'r', 1.5, clutch_start, clutch_end);
legend(ax, {'L-GBM', 'GT'},'Location','northwest');
% title(ax, '$$(a)$$', 'FontSize',40, 'Interpreter','latex', 'Units', 'normalized', 'Position', [0.5, -0.35, 0]);
hold off;

ax = nexttile;
hold on;
ax = plot_tile(ax, t, pred_rf(t), 'b', 1.5, clutch_start, clutch_end);
legend(ax, {'RF', 'GT'},'Location','northwest');
% title(ax, '$$(b)$$', 'FontSize',40, 'Interpreter','latex', 'Units', 'normalized', 'Position', [0.5, -0.35, 0]);
hold off;

t_part = clutch_start(1)-200:clutch_end(1)+200;
offset = t_part(1);
ax = nexttile;
hold on;
ax = plot_tile(ax, t_part, pred_lgbm(t_part), 'r', 1.5, clutch_start(1) - offset, clutch_end(1)- offset);
ax = plot_tile(ax, t_part, pred_rf(t_part), 'b', 1.5, clutch_start(1)- offset, clutch_end(1)- offset);
legend(ax, {'L-GBM',  'GT', 'RF',},'Location','northwest');
% title(ax, '$$(c)$$', 'FontSize',40, 'Interpreter','latex', 'Units', 'normalized', 'Position', [0.5, -0.35, 0]);
hold off;

t_part = clutch_start(2)-100:clutch_end(2)+100;
offset = t_part(1);
ax = nexttile;
hold on;
ax = plot_tile(ax, t_part, pred_lgbm(t_part), 'r', 1.5, clutch_start(2) - offset, clutch_end(2)- offset);
ax = plot_tile(ax, t_part, pred_rf(t_part), 'b', 1.5, clutch_start(2)- offset, clutch_end(2)- offset);
legend(ax, {'L-GBM', 'GT', 'RF'},'Location','northwest');
% title(ax, '$$(d)$$', 'FontSize',40, 'Interpreter','latex', 'Units', 'normalized', 'Position', [0.5, -0.35, 0]);
hold off;

tl.TileSpacing = 'compact';
tl.Padding = 'compact';

%%

function res = inds_wo_clutch(N, box_start, box_end)
inds = 1:N;
clutch_inds = zeros(size(inds));
for i = 1:numel(box_start)
    clutch_inds = clutch_inds | (inds >= box_start(i) & inds <= box_end(i));
end
res = ~clutch_inds;
end

function ax = plot_tile(ax, x, y, color, linewidth, box_start, box_end)
    plot(ax, x, y, "Color", color, 'LineWidth', linewidth);
    % stem(ax, x, y, "Color", color, 'MarkerSize', linewidth);
    if ~isempty(box_start)
        xr = xregion(x(box_start), x(box_end), 'FaceColor', '#FF00FF');
        for i = 1:numel(xr)
            xr(i).FaceAlpha = 0.2;    
        end
    end
    min_y = min(y);
    max_y = max(y);
    center = (max_y + min_y)/2;
    half_range = (max_y - min_y)/2 + (max_y - min_y)/8;
    yticks(ax, [0, 1]);
    ylim(ax, [center-half_range, center+half_range]);
    xlim(ax, [x(1)-100, x(end)+100]);
    ax.FontSize = 25;
end