%%
clear;

%% Load dataset
clc;

addpath(pathdef);

user = 'A';
trial = 1;

s = sftp("sftp://10.150.4.184", "sitl_dvrk", "password", "SITL091520!");
filepath = sprintf("/home/sitl_dvrk/sitl_koh/sitl_dvrk_recs/CRCD/preprocessed/%s_%d", user, trial);
cd(s, filepath);
mget(s, "dvrk_kinematics.bag");

bag = rosbag("dvrk_kinematics.bag");
% glove_tf_bag   = select(bag,'Topic','/vive_tracker_transform');
% psm2_jaw_bag   = select(bag,'Topic','/PSM2/custom/local/setpoint_cp');
% psm1_jaw_bag   = select(bag,'Topic','/PSM1/custom/local/setpoint_cp');

% psm2_local_msg   = readMessages(psm2_local_bag,'DataFormat','struct');
% mtml_local_msg   = readMessages(mtml_local_bag,'DataFormat','struct');
% 
% clear bag psm2_local_bag psm1_jaw_bag;

%%
mput(s,"dvrk_kinematics.bag");
close(s);

%% Extract and organize data

% Number of samples in each bags
full_N = length(glove_tf_msg);

% Load the TFs
t             = zeros(1, full_N);
glove_jaw     = zeros(1, full_N);
psm2_jaw      = zeros(1, full_N);
clutch_flag   = zeros(1, full_N);
glove_cp      = zeros(4, 4, full_N);
glove_raw_cp  = zeros(4, 4, full_N);
psm2_local_cp = zeros(4, 4, full_N);
mtml_local_cp = zeros(4, 4, full_N);

% glove_psm2_offset = vecs2g([0; 0; 1]*pi, [0; 0; 0])*...
%     vecs2g([0; 1; 0]*(-pi/2), [0; 0; 0]);

for i = 1:full_N
    t(i)                 = ext_ros_stamp(glove_raw_tf_msg{i});
    glove_cp(:,:,i)      = transmsg2g(glove_tf_msg{i}.Transform);
    glove_raw_cp(:,:,i)  = transmsg2g(glove_raw_tf_msg{i}.Transform);
    psm2_local_cp(:,:,i) = posestamped2g(psm2_local_msg{i});
    mtml_local_cp(:,:,i) = posestamped2g(mtml_local_msg{i});
    clutch_flag(i)       = clutch_msg{i}.Data;
    glove_jaw(i)         = glove_jaw_msg{i}.Data;
    psm2_jaw(i)          = psm2_jaw_msg{i}.Position;
end

t = sort(t - t(1));

clutch_start = find(diff(clutch_flag)>0);
clutch_end   = find(diff(clutch_flag)<0);

psm2_local_cp_full = psm2_local_cp;

if strcmp(user,'paula') && trial == 4
    N = sum(t<45.13);
elseif strcmp(user,'luciano') && trial == 2
    N = sum(t<57);
else
    N = full_N - 100;
end

if length(clutch_start) == length(clutch_end) + 1
    clutch_start(end) = [];
end

wo_clutch    = inds_wo_clutch(N, clutch_start, clutch_end);

% N = clutch_end(end) + 50;

temp1 = 1:N;
t             = t(temp1);
glove_cp      = glove_cp     (:,:,temp1);
glove_raw_cp  = glove_raw_cp (:,:,temp1);
psm2_local_cp = psm2_local_cp(:,:,temp1);
mtml_local_cp = mtml_local_cp(:,:,temp1);
clutch_flag   = clutch_flag  (temp1);
glove_jaw     = glove_jaw    (temp1);
psm2_jaw      = psm2_jaw     (temp1);

clear glove_tf_msg glove_raw_tf_msg psm2_local_msg mtml_local_msg ...
    clutch_msg glove_jaw_msg psm2_jaw_msg;

%% Extract each position (x, y, z)

glove_cp_tran     = reshape(glove_cp(1:3,4,:),[3,N]);
glove_raw_cp_tran = reshape(glove_raw_cp(1:3,4,:),[3,N]);
psm2_cp_tran      = reshape(psm2_local_cp(1:3,4,:),[3,N]);
psm2_cp_tran_full = reshape(psm2_local_cp_full(1:3,4,:),[3,full_N]);

glove_cp_R     = glove_cp(1:3,1:3,:);
glove_raw_cp_R = glove_raw_cp(1:3,1:3,:);
psm2_cp_R      = psm2_local_cp(1:3,1:3,:);
psm2_cp_R_full = psm2_local_cp_full(1:3,1:3,:);

glove_cp_quat     = compact(quaternion(se3(glove_cp)));
glove_raw_cp_quat = compact(quaternion(se3(glove_raw_cp)));
psm2_cp_quat      = compact(quaternion(se3(psm2_local_cp)));
psm2_cp_quat_full = compact(quaternion(se3(psm2_local_cp_full)));

glove_cp_quat     = clean_quat(glove_cp_quat, 0.3);
glove_raw_cp_quat = clean_quat(glove_raw_cp_quat, 0.3);
psm2_cp_quat      = clean_quat(psm2_cp_quat, 0.3);
psm2_cp_quat_full = clean_quat(psm2_cp_quat_full, 0.3);

%% Plot the trajectory for each axis (Translation)

figure(1);

tl = tiledlayout(3,2);

ax = plot_tile(t, glove_raw_cp_tran(1,:), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{Glove}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{x \ (m)}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_tran(1,:), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{PSM2}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, glove_raw_cp_tran(2,:), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{y}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_tran(2,:), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});

ax = plot_tile(t, glove_raw_cp_tran(3,:), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{z}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_tran(3,:), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');

ylabel(tl, '$$\bf{Translation [x, y, z] (m)}$$', 'FontSize',40, 'Interpreter','latex');
xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize',40, 'Interpreter','latex');

tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%% Plot the trajectory for each axis (Rotation)

figure(2);

tl = tiledlayout(4,2);

ax = plot_tile(t, glove_raw_cp_quat(:,2), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{Glove}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{x}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_quat(:,2), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{PSM2}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, glove_raw_cp_quat(:,3), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{y}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_quat(:,3), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});

ax = plot_tile(t, glove_raw_cp_quat(:,4), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{z}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_quat(:,4), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});

ax = plot_tile(t, glove_raw_cp_quat(:,1), '#EDB120', 1.5, clutch_start, clutch_end, wo_clutch);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_quat(:,1), '#EDB120', 1.5, clutch_start, clutch_end, wo_clutch);

ylabel(tl, '$$\bf{Quaternion [x, y, z, w] (rad)}$$', 'FontSize',40, 'Interpreter','latex');
xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize',40, 'Interpreter','latex');

tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%% Plot the Glove-Jaw Angles

figure(4);

tl = tiledlayout(1,2);

ax = plot_tile(t, glove_jaw, '#A2142F', 1.5, clutch_start, clutch_end, wo_clutch);
title(ax, '$$\bf{Glove}$$', 'FontSize',40, 'Interpreter','latex');
ylabel(ax, '$$\bf{Distance (m)}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_jaw, '#A2142F', 1.5, clutch_start, clutch_end, wo_clutch);
title(ax, '$$\bf{PSM2}$$', 'FontSize',40, 'Interpreter','latex');
ylabel(ax, '$$\bf{Jaw Angle (rad)}$$', 'FontSize',40, 'Interpreter','latex');

xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize',40, 'Interpreter','latex');

tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%%
filename = sprintf('C:\\Users\\kiwil\\Box\\IROS_2024 (Glove)\\Record\\%s.txt', user);
minmax_file_id = fopen(filename,'r');
minmax_file_data = fscanf(minmax_file_id, '%f',[3,4*8])';
min_h = minmax_file_data(4*(trial-1) + 1, :);
max_h = minmax_file_data(4*(trial-1) + 2, :);
min_t = minmax_file_data(4*(trial-1) + 3, :);
max_t = minmax_file_data(4*(trial-1) + 4, :);

%% Rescale PSM2 Pose relative to the glove...

init_psm2_cp_tran = psm2_cp_tran(:,1);
psm2_cp_tran_rescaled = zeros(size(psm2_cp_tran));
psm2_cp_tran_full_rescaled = zeros(size(psm2_cp_tran_full));

% distance
psm2_cp_tran_rescaled(1,:) = inv_scale( ...
    psm2_cp_tran(1,:), min_h(1), max_h(1), ...
    min_t(1), max_t(1));

psm2_cp_tran_rescaled(2,:) = inv_scale( ...
    psm2_cp_tran(2,:), min_h(2), max_h(2), ...
    min_t(2), max_t(2));

psm2_cp_tran_rescaled(3,:) = inv_scale( ...
    psm2_cp_tran(3,:), min_h(3), max_h(3), ...
    min_t(3), max_t(3));

psm2_cp_tran_full_rescaled(1,:) = inv_scale( ...
    psm2_cp_tran_full(1,:), min_h(1), max_h(1), ...
    min_t(1), max_t(1));

psm2_cp_tran_full_rescaled(2,:) = inv_scale( ...
    psm2_cp_tran_full(2,:), min_h(2), max_h(2), ...
    min_t(2), max_t(2));

psm2_cp_tran_full_rescaled(3,:) = inv_scale( ...
    psm2_cp_tran_full(3,:), min_h(3), max_h(3), ...
    min_t(3), max_t(3));

%% Find delay of the signal
clc;
D = zeros(7,1);
for i = 1:height(glove_cp_tran) + width(glove_cp_quat)
    if i <=3
        max_temp = max(glove_cp_tran(i,wo_clutch));
        min_temp = min(glove_cp_tran(i,wo_clutch));
        [~, ~, D(i)] = alignsignals( ...
            glove_cp_tran(i, :), psm2_cp_tran_rescaled(i, :), Method="maxpeak", ...
            MinPeakProminence=0.2*(max_temp-min_temp)....
        );
    else
        max_temp = max(glove_cp_quat(wo_clutch, i-3));
        min_temp = min(glove_cp_quat(wo_clutch, i-3));
        [~, ~, D(i)] = alignsignals( ...
            glove_cp_quat(:, i-3), psm2_cp_quat(:, i-3), Method="maxpeak", ...
            MinPeakProminence=0.2*(max_temp-min_temp)...
        );
    end
end

avg_D = round(mean(rmoutliers(abs(D))), 0);
% avg_D = round(mean(D), 0);

psm2_cp_tran_rescaled_align = zeros(size(psm2_cp_tran_rescaled));
psm2_cp_tran_rescaled_align(:, 1:end-avg_D) = psm2_cp_tran_rescaled(:, avg_D+1:end);
psm2_cp_tran_rescaled_align(:, end-avg_D+1:end) = psm2_cp_tran_full_rescaled(:,N+1:N+avg_D);

psm2_cp_R_align = zeros(size(psm2_cp_R));
psm2_cp_R_align(:,:,1:end-avg_D) = psm2_cp_R(:,:,avg_D+1:end);
psm2_cp_R_align(:,:,end-avg_D+1:end) = psm2_cp_R_full(:,:,N+1:N+avg_D);

psm2_cp_quat_align = zeros(size(psm2_cp_quat));
psm2_cp_quat_align(1:end-avg_D,:) = psm2_cp_quat(avg_D+1:end,:);
psm2_cp_quat_align(end-avg_D+1:end,:) = psm2_cp_quat_full(N+1:N+avg_D,:);

psm2_cp_align = zeros(size(psm2_local_cp));
psm2_cp_align(:,:,1:end-avg_D) = psm2_local_cp(:,:,avg_D+1:end);
psm2_cp_align(:,:,end-avg_D+1:end) = psm2_local_cp_full(:,:,N+1:N+avg_D);

avg_D_sec = (t(end) - t(1))/N*avg_D;

%%

figure(10101);

subplot(2,1,1)
plot(t, glove_cp_tran, t, psm2_cp_tran_rescaled_align);

subplot(2,1,2)
plot(t, glove_cp_quat, t, psm2_cp_quat_align);

%% Plot the trajectory in 3D space

% Interpolate the PSM data

n = 1:N;
k = 4;
nq = linspace(1, N, N*k);
tq = linspace(t(1), t(end), N*k);

psm2_cp_tran_interp = zeros(3, N*k);
psm2_cp_tran_interp(1,:) = interp1(n, psm2_cp_tran_rescaled_align(1,:), nq, 'pchip');
psm2_cp_tran_interp(2,:) = interp1(n, psm2_cp_tran_rescaled_align(2,:), nq, 'pchip');
psm2_cp_tran_interp(3,:) = interp1(n, psm2_cp_tran_rescaled_align(3,:), nq, 'pchip');

figure(3);
n_sub = clutch_end(1):clutch_end(1)+300;
nq_sub = find(abs(nq-n_sub(1))<0.2):find(abs(nq-n_sub(end))<0.2);

tl = tiledlayout(1,2);
ax = nexttile;
h = scatter3(ax, glove_raw_cp_tran(1,n_sub), glove_raw_cp_tran(2,n_sub), ...
    glove_raw_cp_tran(3,n_sub), 50, t(n_sub));
h.MarkerFaceColor = 'flat';
colormap(jet);
grid on;
ax.FontSize = 30;
title(ax, '$$\bf{Glove}$$', 'FontSize',40, 'Interpreter','latex');
xlabel(ax, '$$\bf{x \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
ylabel(ax, '$$\bf{y \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
zlabel(ax, '$$\bf{z \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
min_t = round(min(glove_raw_cp_tran(:,n_sub),[],2), 2);
max_t = round(max(glove_raw_cp_tran(:,n_sub),[],2), 2);
ticks = round((max_t - min_t)/2.5, 2);
xticks(ax, min_t(1):ticks(1):max_t(1));
yticks(ax, min_t(2):ticks(2):max_t(2));
zticks(ax, min_t(3):ticks(3):max_t(3));

ax = nexttile;

h = scatter3(psm2_cp_tran_interp(1,nq_sub), ...
    psm2_cp_tran_interp(2,nq_sub), ...
    psm2_cp_tran_interp(3,nq_sub), 50, tq(nq_sub));
h.MarkerFaceColor = 'flat';
colormap(jet);
colorbar;
grid on;
ax.FontSize = 30;
title(ax, '$$\bf{PSM2}$$', 'FontSize',40, 'Interpreter','latex');
xlabel(ax, '$$\bf{x \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
ylabel(ax, '$$\bf{y \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
zlabel(ax, '$$\bf{z \ (m)}$$', 'FontSize',40, 'Interpreter','latex');
min_t = round(min(psm2_cp_tran_interp(:,nq_sub),[],2), 3);
max_t = round(max(psm2_cp_tran_interp(:,nq_sub),[],2), 3);
ticks = round((max_t - min_t)/2.5, 3);
xticks(ax, min_t(1):ticks(1):max_t(1));
yticks(ax, min_t(2):ticks(2):max_t(2));
zticks(ax, min_t(3):ticks(3):max_t(3));

tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%%

p_dist = vecnorm(glove_cp_tran - psm2_cp_tran_rescaled_align);
R_dist = Rdist(glove_cp_R, psm2_cp_R_align);
q_dist = dist(quaternion(glove_cp_quat), quaternion(psm2_cp_quat_align));
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

min_y = min(glove_cp_tran(1,temp));
max_y = max(glove_cp_tran(1,temp));
ytick_step = round((max_y - min_y)/2.2, 2);
center = (max_y + min_y)/2;
half_range = (max_y - min_y)/2 + (max_y - min_y)/8;

figure(8);

tl = tiledlayout(2,1);

ax = nexttile;

t_temp = t(temp);
plot(ax, t(temp), glove_cp_tran(1,temp), t(temp), psm2_cp_tran_rescaled(1,temp), ...
    'LineWidth', 1.5);
legend({'Glove','PSM Rescaled'}, "Location", 'best', 'FontSize',20);
xticks(ax, round(t_temp(1),0):5:round(t_temp(end),0));
yticks(ax, round(min_y,2):ytick_step:round(max_y,2));
ylim(ax, [center-half_range, center+half_range]);
xlim(ax, [t_temp(1), t_temp(end)+0.1]);
ax.FontSize = 25;

ax = nexttile;
plot(ax, t(temp), glove_cp_tran(1,temp), t(temp), psm2_cp_tran_rescaled_align(1,temp), ...
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

tl = tiledlayout(2,2);

ax = plot_tile(t, p_dist, 'r', 1.5, clutch_start, clutch_end, wo_clutch);
title(ax, '$$\bf{Euclidean}$$', 'FontSize',40, 'Interpreter','latex');
ylabel(ax, '$$\bf{Distance (m)}$$', 'FontSize', 40, 'Interpreter', 'latex');

ax = plot_tile(t, q_dist, 'b', 1.5, clutch_start, clutch_end, wo_clutch);
title(ax, '$$\bf{Quaternion}$$', 'FontSize',40, 'Interpreter','latex');
ylabel(ax, '$$\bf{Distance (rad)}$$', 'FontSize', 40, 'Interpreter', 'latex');

ax = plot_tile(t, R_dist, 'b', 1.5, clutch_start, clutch_end, wo_clutch);
title(ax, '$$\bf{Rotation}$$', 'FontSize',40, 'Interpreter','latex');
ylabel(ax, '$$\bf{Distance (rad)}$$', 'FontSize', 40, 'Interpreter', 'latex');

ax = plot_tile(t, abs(R_dist - q_dist'), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
title(ax, '$$\bf{Quat - Rot}$$', 'FontSize',40, 'Interpreter','latex');
ylabel(ax, '$$\bf{Distance (rad)}$$', 'FontSize', 40, 'Interpreter', 'latex');

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