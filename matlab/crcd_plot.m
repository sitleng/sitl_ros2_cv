%%
clear;

%% Load dataset
clc;

addpath(pathdef);

user = 'B';
trial = 1;
filename = sprintf('C:\\Users\\kiwil\\Box\\CRCD\\%s_%d\\dvrk_kinematics.bag', user, trial);
bag = rosbag(filename);
psm1_bag   = select(bag,'Topic','/PSM1/custom/setpoint_cp');
mtmr_bag   = select(bag,'Topic','/MTMR/measured_cp');

psm1_msg   = readMessages(psm1_bag,'DataFormat','struct');
mtmr_msg   = readMessages(mtmr_bag,'DataFormat','struct');

clear bag psm1_bag mtmr_bag;

%% Extract and organize data

% Number of samples in each bags
full_N = length(psm1_msg);

% Load the TFs
t             = zeros(1, full_N);
psm1_cp = zeros(4, 4, full_N);
mtmr_cp = zeros(4, 4, full_N);

for i = 1:full_N
    t(i)           = ext_ros_stamp(psm1_msg{i});
    psm1_cp(:,:,i) = tfstamped2g(psm1_msg{i});
    mtmr_cp(:,:,i) = tfstamped2g(mtmr_msg{i});
end

t = sort(t - t(1));

clear psm1_msg mtmr_msg;

%% Extract each position (x, y, z)

mtmr_cp_tran = reshape(mtmr_cp(1:3,4,:),[3,full_N]);
psm1_cp_tran = reshape(psm1_cp(1:3,4,:),[3,full_N]);

mtmr_cp_R = mtmr_cp(1:3,1:3,:);
psm1_cp_R = psm1_cp(1:3,1:3,:);

mtmr_cp_quat = compact(quaternion(se3(mtmr_cp)));
psm1_cp_quat = compact(quaternion(se3(psm1_cp)));

mtmr_cp_quat = clean_quat(mtmr_cp_quat, 0.3);
psm1_cp_quat = clean_quat(psm1_cp_quat, 0.3);

%% Plot the trajectory for each axis (Translation)

figure(1);

tl = tiledlayout(3,2);

wo_clutch = 1:full_N;
clutch_start = [];
clutch_end   = [];

ax = plot_tile(t, mtmr_cp_tran(1,:), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{MTMR}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{x \ (m)}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm1_cp_tran(1,:), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{PSM1}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, mtmr_cp_tran(2,:), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{y}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm1_cp_tran(2,:), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});

ax = plot_tile(t, mtmr_cp_tran(3,:), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{z}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm1_cp_tran(3,:), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');

ylabel(tl, '$$\bf{Translation [x, y, z] (m)}$$', 'FontSize',40, 'Interpreter','latex');
xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize',40, 'Interpreter','latex');

tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%% Plot the trajectory for each axis (Rotation)

figure(2);

tl = tiledlayout(4,2);

ax = plot_tile(t, mtmr_cp_quat(:,2), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{Glove}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{x}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm1_cp_quat(:,2), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{PSM2}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, mtmr_cp_quat(:,3), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{y}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm1_cp_quat(:,3), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});

ax = plot_tile(t, mtmr_cp_quat(:,4), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{z}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm1_cp_quat(:,4), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});

ax = plot_tile(t, mtmr_cp_quat(:,1), '#EDB120', 1.5, clutch_start, clutch_end, wo_clutch);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm1_cp_quat(:,1), '#EDB120', 1.5, clutch_start, clutch_end, wo_clutch);

ylabel(tl, '$$\bf{Quaternion [x, y, z, w] (rad)}$$', 'FontSize',40, 'Interpreter','latex');
xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize',40, 'Interpreter','latex');

tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%% Plot the trajectory in 3D space

% Interpolate the PSM data

% n = 1:N;
% k = 4;
% nq = linspace(1, N, N*k);
% tq = linspace(t(1), t(end), N*k);
% 
% psm2_cp_tran_interp = zeros(3, N*k);
% psm2_cp_tran_interp(1,:) = interp1(n, psm2_cp_tran_rescaled_align(1,:), nq, 'pchip');
% psm2_cp_tran_interp(2,:) = interp1(n, psm2_cp_tran_rescaled_align(2,:), nq, 'pchip');
% psm2_cp_tran_interp(3,:) = interp1(n, psm2_cp_tran_rescaled_align(3,:), nq, 'pchip');

figure(3);
tq = t;
temp = 1000;
n_sub = temp:temp+200;
nq_sub = n_sub;
% nq_sub = find(abs(nq-n_sub(1))<0.2):find(abs(nq-n_sub(end))<0.2);

tl = tiledlayout(1,2);
ax = nexttile;
h = scatter3(ax, mtmr_cp_tran(1,n_sub), mtmr_cp_tran(2,n_sub), ...
    mtmr_cp_tran(3,n_sub), 50, t(n_sub));
h.MarkerFaceColor = 'flat';
colormap(jet);
grid on;
ax.FontSize = 30;
title(ax, '$$\bf{MTMR}$$', 'FontSize',40, 'Interpreter','latex');
% xlabel(ax, '$$\bf{x \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{y \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
% zlabel(ax, '$$\bf{z \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
set(gca,'XTick',[])
set(gca,'YTick',[])
set(gca,'ZTick',[])

ax = nexttile;

h = scatter3(psm1_cp_tran(1,nq_sub), ...
    psm1_cp_tran(2,nq_sub), ...
    psm1_cp_tran(3,nq_sub), 50, tq(nq_sub));
h.MarkerFaceColor = 'flat';
% colormap(jet);
% colorbar;
grid on;
ax.FontSize = 30;
title(ax, '$$\bf{PSM2}$$', 'FontSize',40, 'Interpreter','latex');
% xlabel(ax, '$$\bf{x \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{y \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
% zlabel(ax, '$$\bf{z \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
set(gca,'XTick',[])
set(gca,'YTick',[])
set(gca,'ZTick',[])

tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%%

p_dist = vecnorm(mtmr_cp_tran - psm2_cp_tran_rescaled_align);
R_dist = Rdist(mtmr_cp_R, psm2_cp_R_align);
q_dist = dist(quaternion(mtmr_cp_quat), quaternion(psm2_cp_quat_align));
% q_dist = dist(quaternion(se3(glove_cp)), quaternion(se3(psm2_cp_align)));

mean_p_dist = mean(p_dist(:,wo_clutch));
std_p_dist = std(p_dist(:,wo_clutch));

mean_R_dist = mean(R_dist(:,wo_clutch));
std_R_dist = std(R_dist(:,wo_clutch));

mean_q_dist = mean(q_dist(wo_clutch,:));
std_q_dist = std(q_dist(wo_clutch,:));

%%
clc;

fprintf('Dataset: %s %d\n', user, trial);

fprintf('Duration: %.3f (s)\n', t(end) - t(1));

fprintf('Delay: %.3f (s)\n', avg_D_sec);

fprintf('P Distance: \n Average:   %.3f \n Std. Dev.: %.3f \n', mean_p_dist, std_p_dist);

fprintf('R Distance (Quaternion): \n Average:   %.3f \n Std. Dev.: %.3f \n', mean_q_dist, std_q_dist);

fprintf('R Distance (Rotation): \n Average:   %.3f \n Std. Dev.: %.3f \n', mean_R_dist, std_R_dist);

%%
list_p_dist = cat(2, list_p_dist, p_dist(:,wo_clutch));
list_R_dist = cat(2, list_R_dist, R_dist(:,wo_clutch));
list_q_dist = cat(1, list_q_dist, q_dist(wo_clutch,:));

%%
clc;

fprintf('Dataset: %s\n', user);

fprintf('Overall P Distance: \n Average:   %.3f \n Std. Dev.: %.3f \n', mean(list_p_dist), std(list_p_dist));

fprintf('Overall R Distance (Quaternion): \n Average:   %.3f \n Std. Dev.: %.3f \n', mean(list_q_dist), std(list_q_dist));

fprintf('Overall R Distance (Rotation): \n Average:   %.3f \n Std. Dev.: %.3f \n', mean(list_R_dist), std(list_R_dist));

%% Display the delay of the signals, and the shifted signal.

if ~isempty(clutch_start)
    temp = 1:clutch_start(1);
else
    temp = 1:N;
end

min_y = min(mtmr_cp_tran(1,temp));
max_y = max(mtmr_cp_tran(1,temp));
ytick_step = round((max_y - min_y)/2.2, 2);
center = (max_y + min_y)/2;
half_range = (max_y - min_y)/2 + (max_y - min_y)/8;

figure(8);

tl = tiledlayout(2,1);

ax = nexttile;

t_temp = t(temp);
plot(ax, t(temp), mtmr_cp_tran(1,temp), t(temp), psm2_cp_tran_rescaled(1,temp), ...
    'LineWidth', 1.5);
legend({'Glove','PSM Rescaled'}, "Location", 'best', 'FontSize',20);
xticks(ax, round(t_temp(1),0):5:round(t_temp(end),0));
yticks(ax, round(min_y,2):ytick_step:round(max_y,2));
ylim(ax, [center-half_range, center+half_range]);
xlim(ax, [t_temp(1), t_temp(end)+0.1]);
ax.FontSize = 25;

ax = nexttile;
plot(ax, t(temp), mtmr_cp_tran(1,temp), t(temp), psm2_cp_tran_rescaled_align(1,temp), ...
    'LineWidth', 1.5);
legend({'Glove','PSM Rescaled'}, "Location", 'best', 'FontSize',20);
xticks(ax, round(t_temp(1),0):5:round(t_temp(end),0));
yticks(ax, round(min_y,2):ytick_step:round(max_y,2));
ylim(ax, [center-half_range, center+half_range]);
xlim(ax, [t_temp(1), t_temp(end)+0.1]);
ax.FontSize = 25;

ylabel(tl, '$$\bf{Distance (m)}$$', 'FontSize', 30, 'Interpreter', 'latex');
xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize', 30, 'Interpreter', 'latex');

%%

exportgraphics(gcf, "C:\\Users\\kiwil\\Box\\IROS_2024 (Glove)\\glove_psm_delay.pdf")

%%

figure(6);

% tl = tiledlayout(2,2);
tl = tiledlayout(1,2);

ax = plot_tile(t, p_dist, 'r', 1.5, clutch_start, clutch_end, wo_clutch);
title(ax, '$$\bf{Translational \ Error}$$', 'FontSize',40, 'Interpreter','latex');
ylabel(ax, '$$\bf{Distance (m)}$$', 'FontSize', 40, 'Interpreter', 'latex');

% ax = plot_tile(t, q_dist, 'b', 1.5, clutch_start, clutch_end, wo_clutch);
% title(ax, '$$\bf{Quaternion}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{Distance (rad)}$$', 'FontSize', 40, 'Interpreter', 'latex');

ax = plot_tile(t, R_dist, 'b', 1.5, clutch_start, clutch_end, wo_clutch);
title(ax, '$$\bf{Rotational \ Error}$$', 'FontSize',40, 'Interpreter','latex');
ylabel(ax, '$$\bf{Distance (rad)}$$', 'FontSize', 40, 'Interpreter', 'latex');

% ax = plot_tile(t, abs(R_dist - q_dist'), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
% title(ax, '$$\bf{Quat - Rot}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{Distance (rad)}$$', 'FontSize', 40, 'Interpreter', 'latex');

xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize', 40, 'Interpreter', 'latex');

%% Plot Functions

function orig_x = inv_scale(new_x, min_o, max_o, min_n, max_n)
orig_x = (new_x-min_n)*(max_o-min_o)/(max_n-min_n) + min_o;
end

function res = inds_wo_clutch(N, box_start, box_end)
inds = 1:N;
clutch_inds = zeros(size(inds));
for i = 1:numel(box_start)
    clutch_inds = clutch_inds | (inds >= box_start(i) & inds <= box_end(i));
end
res = ~clutch_inds;
end

function ax = plot_tile(x, y, color, linewidth, box_start, box_end, wo_clutch)
    ax = nexttile;
    plot(ax, x, y, "Color", color, 'LineWidth', linewidth);
    hold on;
    if ~isempty(box_start)
        xr = xregion(x(box_start), x(box_end), 'FaceColor', '#FF00FF');
        for i = 1:numel(xr)
            xr(i).FaceAlpha = 0.1;
        end
    end
    xticks(ax, round(x(1),0):10:round(x(end),0));
    hold off;
    y_wo_clutch = y(wo_clutch);
    min_y = min(y_wo_clutch);
    max_y = max(y_wo_clutch);
    ytick_step = round((max_y - min_y)/2.2, 3);
    center = (max_y + min_y)/2;
    half_range = (max_y - min_y)/2 + (max_y - min_y)/8;
    yticks(ax, round(min_y,2):ytick_step:round(max_y,2));
    ylim(ax, [center-half_range, center+half_range]);
    xlim(ax, [x(1), x(end)+0.1]);
    ax.FontSize = 25;
end