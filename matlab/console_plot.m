%%
clear; clc;

%%

list_p_dist = [];
list_R_dist = [];
list_q_dist = [];

%%

ext_users = {'alvaro', 'diego', 'valentina', 'mhd'};

%% Load dataset

addpath(pathdef);

user = 'arman';
trial = 5;

if any(strcmp(user,ext_users))
    filename = sprintf('C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\icra_2025\\glove_console_%s%d_ext.bag', user, trial);
else
    filename = sprintf('C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\icra_2025\\glove_console_%s%d.bag', user, trial);
end

bag = rosbag(filename);

mtml_bag   = select(bag,'Topic','/MTML/measured_cp');
% psm2_bag = select(bag,'Topic','/PSM2/custom/setpoint_cp');
psm2_bag   = select(bag,'Topic','/PSM2/measured_cp');
clutch_bag = select(bag,'Topic','/pedals/clutch');

mtml_msg   = readMessages(mtml_bag, 'DataFormat','struct');
psm2_msg   = readMessages(psm2_bag,'DataFormat','struct');
clutch_msg = readMessages(clutch_bag,'DataFormat','struct');

clear mtml_bag psm2_bag clutch_bag;

%% Extract and organize data

% Number of samples in each bags
full_N = length(mtml_msg);

% Load the TFs
t           = zeros(1, full_N);
mtml_cp     = zeros(4, 4, full_N);
psm2_cp     = zeros(4, 4, full_N);
clutch_flag = zeros(1, full_N);

for i = 1:full_N
    t(i)           = ext_ros_stamp(mtml_msg{i});
    mtml_cp(:,:,i) = posestamped2g(mtml_msg{i});
    psm2_cp(:,:,i) = posestamped2g(psm2_msg{i});
    clutch_flag(i) = clutch_msg{i}.Data;
end

% Scale time
t = sort(t - t(1));

% Find the moments when the clutch was activated and released
clutch_start = find(diff(clutch_flag)>0);
clutch_end   = find(diff(clutch_flag)<0);
if length(clutch_start) == length(clutch_end) + 1
    clutch_start(end) = [];
end

psm2_cp_full = psm2_cp;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(user,'diego') && trial == 6
    temp1 = find(t>315);
elseif strcmp(user,'paula') && trial == 6
    temp1 = 1:clutch_start(end);
    clutch_start = clutch_start(1:end-1);
    clutch_end   = clutch_end(1:end-1);
else
    temp1 = 1:full_N - 50;
end
N = length(temp1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wo_clutch    = inds_wo_clutch(temp1, clutch_start, clutch_end);

if ~isempty(clutch_start)
    clutch_start = find(ismember(temp1, clutch_start));
end
if ~isempty(clutch_end)
    clutch_end   = find(ismember(temp1, clutch_end));
end

t       = t(temp1);
mtml_cp = mtml_cp(:,:,temp1);
psm2_cp = psm2_cp(:,:,temp1);

clear mtml_msg psm2_local_msg clutch_msg;

%% Extract each position (x, y, z)

mtml_cp_tran      = reshape(mtml_cp(1:3,4,:),[3,N]);
psm2_cp_tran      = reshape(psm2_cp(1:3,4,:),[3,N]);
psm2_cp_tran_full = reshape(psm2_cp_full(1:3,4,:),[3,full_N]);

mtml_cp_R         = mtml_cp(1:3,1:3,:);
psm2_cp_R         = psm2_cp(1:3,1:3,:);
psm2_cp_R_full    = psm2_cp_full(1:3,1:3,:);

mtml_cp_quat      = compact(quaternion(se3(mtml_cp)));
psm2_cp_quat      = compact(quaternion(se3(psm2_cp)));
psm2_cp_quat_full = compact(quaternion(se3(psm2_cp_full)));

mtml_cp_quat      = clean_quat(mtml_cp_quat, 0.1);
psm2_cp_quat      = clean_quat(psm2_cp_quat, 0.1);
psm2_cp_quat_full = clean_quat(psm2_cp_quat_full, 0.1);

%% Plot the trajectory for each axis (Translation)


figure(1);

tl = tiledlayout(3,2);

ax = plot_tile(t, mtml_cp_tran(1,:), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{Glove}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{x \ (m)}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_tran(1,:), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{PSM2}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, mtml_cp_tran(2,:), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{y}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_tran(2,:), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});

ax = plot_tile(t, mtml_cp_tran(3,:), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
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

ax = plot_tile(t, mtml_cp_quat(:,2), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{Glove}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{x}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_quat(:,2), 'r', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
title(ax, '$$\bf{PSM2}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, mtml_cp_quat(:,3), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{y}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_quat(:,3), 'g', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});

ax = plot_tile(t, mtml_cp_quat(:,4), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{z}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile(t, psm2_cp_quat(:,4), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
xticklabels(ax,{});

plot_tile(t, mtml_cp_quat(:,1), '#EDB120', 1.5, clutch_start, clutch_end, wo_clutch);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');

plot_tile(t, psm2_cp_quat(:,1), '#EDB120', 1.5, clutch_start, clutch_end, wo_clutch);

ylabel(tl, '$$\bf{Quaternion [x, y, z, w] (rad)}$$', 'FontSize',40, 'Interpreter','latex');
xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize',40, 'Interpreter','latex');

tl.Padding = 'compact';
tl.TileSpacing = 'compact';




%% Rescale PSM2 Pose relative to the mtml...

scale = 1/0.2;

% distance
psm2_cp_tran_rescaled = psm2_cp_tran*scale;

psm2_cp_tran_full_rescaled = psm2_cp_tran_full*scale;

%%

cp_tran_offset = zeros(3, N);
cp_quat_offset = zeros(N, 4);
cp_quat_offset(:,1) = 1;

pt_starts = [1, clutch_end];
pt_ends   = [clutch_start, N];

for i = 1:length(pt_starts)
    tran_offset = mean(mtml_cp_tran(:, pt_starts(i):pt_ends(i)) - psm2_cp_tran_rescaled(:, pt_starts(i):pt_ends(i)), 2);
    quat_offset = compact(meanrot(quaternion(quatmultiply(mtml_cp_quat(pt_starts(i):pt_ends(i), :), quatinv(psm2_cp_quat(pt_starts(i):pt_ends(i), :))))));
    cp_tran_offset(:, pt_starts(i):pt_ends(i)) = repmat(tran_offset, 1, pt_ends(i)-pt_starts(i) + 1);
    cp_quat_offset(pt_starts(i):pt_ends(i), :) = repmat(quat_offset, pt_ends(i)-pt_starts(i) + 1, 1);
end

psm2_cp_tran_rescaled = psm2_cp_tran_rescaled + cp_tran_offset;
psm2_cp_quat = quatmultiply(cp_quat_offset, psm2_cp_quat);

%% Find delay of the signal
D = zeros(7,1);
for i = 1:height(mtml_cp_tran) + width(mtml_cp_quat)
    if i <=3
        max_temp = max(mtml_cp_tran(i,wo_clutch));
        min_temp = min(mtml_cp_tran(i,wo_clutch));
        % [~, ~, D(i)] = alignsignals( ...
        %     mtml_cp_tran(i, :), psm2_cp_tran_rescaled(i, :), Method="maxpeak", ...
        %     MinPeakProminence=0.5*(max_temp-min_temp)....
        % );
        [~, ~, D(i)] = alignsignals( ...
            mtml_cp_tran(i, :), psm2_cp_tran_rescaled(i, :), Method="xcorr");
    else
        max_temp = max(mtml_cp_quat(wo_clutch, i-3));
        min_temp = min(mtml_cp_quat(wo_clutch, i-3));
        % [~, ~, D(i)] = alignsignals( ...
        %     mtml_cp_quat(:, i-3), psm2_cp_quat(:, i-3), Method="maxpeak", ...
        %     MinPeakProminence=0.4*(max_temp-min_temp)...
        % );
        [~, ~, D(i)] = alignsignals( ...
            mtml_cp_quat(:, i-3), psm2_cp_quat(:, i-3), Method="xcorr");
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

psm2_cp_align = zeros(size(psm2_cp));
psm2_cp_align(:,:,1:end-avg_D) = psm2_cp(:,:,avg_D+1:end);
psm2_cp_align(:,:,end-avg_D+1:end) = psm2_cp_full(:,:,N+1:N+avg_D);

avg_D_sec = (t(end) - t(1))/N*avg_D;


%%
figure(10101);

subplot(2,1,1)
plot(t, mtml_cp_tran, t, psm2_cp_tran_rescaled_align);

subplot(2,1,2)
plot(t, mtml_cp_quat, t, psm2_cp_quat_align);

legend;

%%

p_dist = vecnorm(mtml_cp_tran - psm2_cp_tran_rescaled_align);
R_dist = Rdist(mtml_cp_R, psm2_cp_R_align);
q_dist = dist(quaternion(mtml_cp_quat), quaternion(psm2_cp_quat_align));

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

min_y = min(mtml_cp_tran(1,temp));
max_y = max(mtml_cp_tran(1,temp));
ytick_step = round((max_y - min_y)/2.2, 2);
center = (max_y + min_y)/2;
half_range = (max_y - min_y)/2 + (max_y - min_y)/8;

figure(8);

tl = tiledlayout(2,1);

ax = nexttile;

t_temp = t(temp);
plot(ax, t(temp), mtml_cp_tran(1,temp), t(temp), psm2_cp_tran_rescaled(1,temp), ...
    'LineWidth', 1.5);
legend({'Glove','PSM Rescaled'}, "Location", 'best', 'FontSize',20);
xticks(ax, round(t_temp(1),0):5:round(t_temp(end),0));
yticks(ax, round(min_y,2):ytick_step:round(max_y,2));
ylim(ax, [center-half_range, center+half_range]);
xlim(ax, [t_temp(1), t_temp(end)+0.1]);
ax.FontSize = 25;

ax = nexttile;
plot(ax, t(temp), mtml_cp_tran(1,temp), t(temp), psm2_cp_tran_rescaled_align(1,temp), ...
    'LineWidth', 1.5);
legend({'Glove','PSM Rescaled'}, "Location", 'best', 'FontSize',20);
xticks(ax, round(t_temp(1),0):5:round(t_temp(end),0));
yticks(ax, round(min_y,2):ytick_step:round(max_y,2));
ylim(ax, [center-half_range, center+half_range]);
xlim(ax, [t_temp(1), t_temp(end)+0.1]);
ax.FontSize = 25;

ylabel(tl, '$$\bf{Distance (m)}$$', 'FontSize', 30, 'Interpreter', 'latex');
xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize', 30, 'Interpreter', 'latex');

%% Save Glove-PSM Delay Plot

exportgraphics(gcf, "C:\\Users\\kiwil\\Box\\IROS_2024 (Glove)\\mtml_psm_delay.pdf")

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
