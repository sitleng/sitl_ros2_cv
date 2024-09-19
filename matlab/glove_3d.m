%%
clear;
clc;

%%

% Specify the path to your ROS bag file
user = 'arman';
trial = 5;
% is_on_off = false;
% filename = sprintf('C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\icra_2025\\%s_%d.bag', user, trial);

is_on_off = true;
filename = 'C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\icra_2025\\icra_video_3.bag';

% Create a rosbag object
bag = rosbag(filename);

% List the available topics in the bag
disp('Available Topics:');
disp(bag.AvailableTopics);

%% Load dataset

glove_tf_bag     = select(bag,'Topic','/tracker_current_pos_tf');
glove_raw_tf_bag = select(bag,'Topic','/tracker_current_raw_data');
psm2_local_bag   = select(bag,'Topic','/PSM2/custom/local/setpoint_cp');
clutch_bag       = select(bag,'Topic','glove/left/fist');

glove_tf_msg     = readMessages(glove_tf_bag, 'DataFormat','struct');
glove_raw_tf_msg = readMessages(glove_raw_tf_bag, 'DataFormat','struct');
psm2_local_msg   = readMessages(psm2_local_bag,'DataFormat','struct');
clutch_msg       = readMessages(clutch_bag,'DataFormat','struct');

if is_on_off
    on_off_bag       = select(bag,'Topic','/glove/left/on_off');
    on_off_msg       = readMessages(on_off_bag,'DataFormat','struct');
    clear on_off_bag;
end

clear bag glove_tf_bag glove_raw_tf_bag psm2_local_bag clutch_bag;

%% Extract and organize data

% Number of samples in each bags
full_N = length(glove_raw_tf_msg);

% Load the TFs
t             = zeros(1, full_N);
glove_cp      = zeros(4, 4, full_N);
glove_raw_cp  = zeros(4, 4, full_N);
psm2_local_cp = zeros(4, 4, full_N);
clutch_flag   = zeros(1, full_N);
on_off_flag   = zeros(1, full_N);

for i = 1:full_N
    t(i)                 = ext_ros_stamp(glove_raw_tf_msg{i});
    glove_cp(:,:,i)      = transmsg2g(glove_tf_msg{i}.Transform);
    glove_raw_cp(:,:,i)  = transmsg2g(glove_raw_tf_msg{i}.Transform);
    psm2_local_cp(:,:,i) = posestamped2g(psm2_local_msg{i});
    clutch_flag(i)       = clutch_msg{i}.Data;
    if is_on_off
        on_off_flag(i)   = on_off_msg{i}.Data;
    end
end

% Scale time
t = sort(t - t(1));

% Find the moments when the clutch was activated and released
clutch_start = find(diff(clutch_flag)>0);
clutch_end   = find(diff(clutch_flag)<0);

% Save a copy of the full version for some signals
t_full = t;
psm2_local_cp_full = psm2_local_cp;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(user,'diego') && trial == 1
    temp1 = find(t>315);
elseif strcmp(user,'paula') && trial == 4
    temp1 = 1:clutch_start(end);
    clutch_start = clutch_start(1:end-1);
    clutch_end   = clutch_end(1:end-1);
elseif is_on_off
    temp1 = 1:full_N-15;
else
    temp1 = 1:full_N-15;
end
N = length(temp1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if length(clutch_start) == length(clutch_end) + 1
    clutch_start(end) = [];
end

wo_clutch    = inds_wo_clutch(temp1, clutch_start, clutch_end);

if ~isempty(clutch_start)
    clutch_start = find(ismember(temp1, clutch_start));
end
if ~isempty(clutch_end)
    clutch_end   = find(ismember(temp1, clutch_end));
end

t             = t(temp1);
glove_cp      = glove_cp     (:,:,temp1);
glove_raw_cp  = glove_raw_cp (:,:,temp1);
psm2_local_cp = psm2_local_cp(:,:,temp1);
if is_on_off
    on_off_start  = find(diff(on_off_flag)>0);
    on_off_end    = find(abs(t - (t(on_off_start) + 3)) <= 1e-2, 1);
    clear on_off_msg;
end

clear glove_tf_msg glove_raw_tf_msg psm2_local_msg clutch_msg;

%% Get the translations of the glove and the PSM2

glove_cp_tran     = reshape(glove_cp(1:3,4,:),[3,N]);
glove_raw_cp_tran = reshape(glove_raw_cp(1:3,4,:),[3,N]);
psm2_cp_tran      = reshape(psm2_local_cp(1:3,4,:),[3,N]);
psm2_cp_tran_full = reshape(psm2_local_cp_full(1:3,4,:),[3,full_N]);

glove_cp_quat     = compact(quaternion(se3(glove_cp)));
glove_raw_cp_quat = compact(quaternion(se3(glove_raw_cp)));
psm2_cp_quat      = compact(quaternion(se3(psm2_local_cp)));
psm2_cp_quat_full = compact(quaternion(se3(psm2_local_cp_full)));

eps = 0.1;
glove_cp_quat     = clean_quat(glove_cp_quat, eps);
glove_raw_cp_quat = clean_quat(glove_raw_cp_quat, eps);
psm2_cp_quat      = clean_quat(psm2_cp_quat, eps);
psm2_cp_quat_full = clean_quat(psm2_cp_quat_full, eps);

%% Load Min_Max for rescaling
% filename = sprintf('C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\min_max\\min_max_%s_%d.txt', user, trial);
filename = 'C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\min_max\\min_max_icra_video.txt';

minmax_file_id = fopen(filename, 'r');

minmax_file_data = {};
while ~feof(minmax_file_id)
   cur_line = fgetl(minmax_file_id);
   if isempty(cur_line)
       continue;
   end
   cur_data = regexp(cur_line, '[-+]?\d+\.?\d*', 'match');
   minmax_file_data = cat(1, minmax_file_data, cellfun(@str2double, cur_data));
end
fclose(minmax_file_id);
min_h = minmax_file_data{1, :};
max_h = minmax_file_data{2, :};
min_t = minmax_file_data{3, :};
max_t = minmax_file_data{4, :};

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

psm2_cp_tran_rescaled_align = zeros(size(psm2_cp_tran_rescaled));
psm2_cp_tran_rescaled_align(:, 1:end-avg_D) = psm2_cp_tran_rescaled(:, avg_D+1:end);
psm2_cp_tran_rescaled_align(:, end-avg_D+1:end) = psm2_cp_tran_full_rescaled(:,temp1(end)+1:temp1(end)+avg_D);

psm2_cp_quat_align = zeros(size(psm2_cp_quat));
psm2_cp_quat_align(1:end-avg_D,:) = psm2_cp_quat(avg_D+1:end,:);
psm2_cp_quat_align(end-avg_D+1:end,:) = psm2_cp_quat_full(temp1(end)+1:temp1(end)+avg_D,:);

psm2_cp_align = zeros(size(psm2_local_cp));
psm2_cp_align(:,:,1:end-avg_D) = psm2_local_cp(:,:,avg_D+1:end);
psm2_cp_align(:,:,end-avg_D+1:end) = psm2_local_cp_full(:,:,temp1(end)+1:temp1(end)+avg_D);

avg_D_sec = (t(end) - t(1))/N*avg_D;

%%

figure(10101);

subplot(2,1,1)
plot(t, glove_cp_tran, t, psm2_cp_tran_rescaled_align);

subplot(2,1,2)
plot(t, glove_cp_quat, t, psm2_cp_quat_align);

%% Find travel distance

clc;

fprintf('Dataset: %s %d\n', user, trial);

fprintf('Duration: %.3f (s)\n', t(end) - t(1));

fprintf('Travel Distance of PSM (Glove): %.3f (m)\n', travel_dist(psm2_cp_tran));

%% Find the limits

limit_offset = 1e-3;

glove_trans_min = min(glove_raw_cp_tran, [], 2);
glove_quat_min  = min(glove_raw_cp_quat);
psm2_trans_min  = min(psm2_cp_tran_rescaled_align, [], 2);
psm2_quat_min   = min(psm2_cp_quat_align);

glove_trans_max = max(glove_raw_cp_tran, [], 2);
glove_quat_max  = max(glove_raw_cp_quat);
psm2_trans_max  = max(psm2_cp_tran_rescaled_align, [], 2);
psm2_quat_max   = max(psm2_cp_quat_align);

%% Set Monitor Position (When using secondary monitor)

mp = get(0, 'MonitorPositions');
des_monitor = 2; % Choose which monitor you would like
set(0, 'defaultFigurePosition', mp(des_monitor,:));

%% Animated 3D Plot

% Initialize the figure

figure('WindowState','fullscreen');

dot_size = 10;
marker = '.';
glove_ax = subplot(1,2,2);
glove_an = animatedline(glove_ax, 'Marker', marker, 'MarkerSize', dot_size);
psm2_ax = subplot(1,2,1);
psm2_an = animatedline(psm2_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', 'b');

hold([glove_ax psm2_ax], 'on');
grid([glove_ax psm2_ax], 'on');

% Initial view
azimuth = 37.5;
elevation = 30;
view(glove_ax, azimuth, elevation);
view(psm2_ax, -azimuth, elevation);

% Labels, legend, and grid
fontsize_label = 25;
fontsize_title  = 24;

glove_ax.FontSize = fontsize_title;
title(glove_ax, 'Glove', 'FontSize', fontsize_label);
xlabel(glove_ax, 'X-axis','FontSize', fontsize_label);
ylabel(glove_ax, 'Y-axis','FontSize', fontsize_label);
zlabel(glove_ax, 'Z-axis','FontSize', fontsize_label);
xlim(glove_ax, [glove_trans_min(1)-limit_offset, glove_trans_max(1)+limit_offset]);
ylim(glove_ax, [glove_trans_min(2)-limit_offset, glove_trans_max(2)+limit_offset]);
zlim(glove_ax, [glove_trans_min(3)-limit_offset, glove_trans_max(3)+limit_offset]);

psm2_ax.FontSize = fontsize_title;
title(psm2_ax, 'PSM', 'FontSize', fontsize_label);
xlabel(psm2_ax, 'X-axis','FontSize', fontsize_label);
ylabel(psm2_ax, 'Y-axis','FontSize', fontsize_label);
zlabel(psm2_ax, 'Z-axis','FontSize', fontsize_label);
xlim(psm2_ax, [psm2_trans_min(1)-limit_offset, psm2_trans_max(1)+limit_offset]);
ylim(psm2_ax, [psm2_trans_min(2)-limit_offset, psm2_trans_max(2)+limit_offset]);
zlim(psm2_ax, [psm2_trans_min(3)-limit_offset, psm2_trans_max(3)+limit_offset]);

clutch_N = length(clutch_start);

colors = cell(size(t));
colors(:) = {'g'};

for i = 1:N-1
    for k = 1:clutch_N
        if i >= clutch_start(k) && i <= clutch_end(k)
            colors{i} = 'r';
        end
    end
    if i >= on_off_start && i <= on_off_end
        colors{i} = 'y';
    elseif i < on_off_start
        colors{i} = [.7 .7 .7];
    end
end

t_diff = diff(t);

% Animation loop for the trajectory
t_start_overall = tic;
for i = 1:N-1
    t_start_it = tic;
    glove_an.Color = colors{i};
    addpoints(glove_an, glove_raw_cp_tran(1,i), glove_raw_cp_tran(2,i), glove_raw_cp_tran(3,i));
    addpoints(psm2_an, psm2_cp_tran_rescaled_align(1,i), psm2_cp_tran_rescaled_align(2,i), psm2_cp_tran_rescaled_align(3,i));
    t_end_it = toc(t_start_it);
    pause(t_diff(i) - t_end_it);
end
t_end_overall = toc(t_start_overall);
fprintf("Elapsed time: %f (s)\n", t_end_overall);

% Plot the ending point
hold([glove_ax psm2_ax], 'off');

%% Animated 2D Plot

% Initialize the figure

figure('WindowState','fullscreen');

dot_size = 10;
marker = '.';

psm2_trans_ax = subplot(2,2,1);
psm2_trans_an_x = animatedline(psm2_trans_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#0072BD");
psm2_trans_an_y = animatedline(psm2_trans_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#D95319");
psm2_trans_an_z = animatedline(psm2_trans_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#EDB120");

psm2_quat_ax = subplot(2,2,3);
psm2_quat_an_w = animatedline(psm2_quat_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#0072BD");
psm2_quat_an_x = animatedline(psm2_quat_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#D95319");
psm2_quat_an_y = animatedline(psm2_quat_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#EDB120");
psm2_quat_an_z = animatedline(psm2_quat_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#7E2F8E");

glove_trans_ax = subplot(2,2,2);
glove_trans_an_x = animatedline(glove_trans_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#0072BD");
glove_trans_an_y = animatedline(glove_trans_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#D95319");
glove_trans_an_z = animatedline(glove_trans_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#EDB120");

glove_quat_ax = subplot(2,2,4);
glove_quat_an_w = animatedline(glove_quat_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#0072BD");
glove_quat_an_x = animatedline(glove_quat_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#D95319");
glove_quat_an_y = animatedline(glove_quat_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#EDB120");
glove_quat_an_z = animatedline(glove_quat_ax, 'Marker', marker, 'MarkerSize', dot_size, 'Color', "#7E2F8E");

ax_list = [glove_trans_ax glove_quat_ax psm2_trans_ax psm2_quat_ax];
hold(ax_list, 'on');
grid(ax_list, 'on');

% Clutch
for j = 1:length(ax_list)
    xr = xregion(ax_list(j), t(clutch_start), t(clutch_end), 'FaceColor', '#FF00FF');
    for i = 1:numel(xr)
        xr(i).FaceAlpha = 0.1;
    end
    
    % On-Off
    xr = xregion(ax_list(j), t(on_off_start), t(on_off_end), 'FaceColor', '#77AC30');
    for i = 1:numel(xr)
        xr(i).FaceAlpha = 0.1;
    end
end

% Labels, legend, and grid
fontsize_label = 25;
fontsize_title  = 24;

glove_trans_ax.FontSize = fontsize_title;
title(glove_trans_ax, 'Glove', 'FontSize', fontsize_label);
xlabel(glove_trans_ax, 'Time (s)','FontSize', fontsize_label);
ylabel(glove_trans_ax, 'Translation (m)','FontSize', fontsize_label);
xlim(glove_trans_ax, [0, t(end)+limit_offset]);
ylim(glove_trans_ax, [min(glove_trans_min)-limit_offset, max(glove_trans_max)+limit_offset]);

glove_quat_ax.FontSize = fontsize_title;
title(glove_quat_ax, 'Glove', 'FontSize', fontsize_label);
xlabel(glove_quat_ax, 'Time (s)','FontSize', fontsize_label);
ylabel(glove_quat_ax, 'Orientation (rads)','FontSize', fontsize_label);
xlim(glove_quat_ax, [0, t(end)+limit_offset]);
ylim(glove_quat_ax, [min(glove_quat_min)-limit_offset, max(glove_quat_max)+limit_offset]);

psm2_trans_ax.FontSize = fontsize_title;
title(psm2_trans_ax, 'PSM', 'FontSize', fontsize_label);
xlabel(psm2_trans_ax, 'Time (s)','FontSize', fontsize_label);
ylabel(psm2_trans_ax, 'Translation (m)','FontSize', fontsize_label);
xlim(psm2_trans_ax, [0, t(end)+limit_offset]);
ylim(psm2_trans_ax, [min(psm2_trans_min)-limit_offset, max(psm2_trans_max)+limit_offset]);

psm2_quat_ax.FontSize = fontsize_title;
title(psm2_quat_ax, 'PSM', 'FontSize', fontsize_label);
xlabel(psm2_quat_ax, 'Time (s)','FontSize', fontsize_label);
ylabel(psm2_quat_ax, 'Orientation (rads)','FontSize', fontsize_label);
xlim(psm2_quat_ax, [0, t(end)+limit_offset]);
ylim(psm2_quat_ax, [min(psm2_quat_min)-limit_offset, max(psm2_quat_max)+limit_offset]);

clutch_N = length(clutch_start);

pause_time = 1.5e-2;

colors = cell(size(t));
colors(:) = {'g'};

for i = 1:N-1
    for k = 1:clutch_N
        if i >= clutch_start(k) && i <= clutch_end(k)
            colors{i} = 'r';
        end
    end
    if i >= on_off_start && i <= on_off_end
        colors{i} = 'y';
    elseif i < on_off_start
        colors{i} = [.7 .7 .7];
    end
end

t_diff = diff(t);

% Animation loop for the trajectory
t_start_overall = tic;
for i = 1:N-1
    t_start_it = tic;
    addpoints(glove_trans_an_x, t(i), glove_raw_cp_tran(1,i));
    addpoints(glove_trans_an_y, t(i), glove_raw_cp_tran(2,i));
    addpoints(glove_trans_an_z, t(i), glove_raw_cp_tran(3,i));

    addpoints(glove_quat_an_w, t(i), glove_raw_cp_quat(i,1));
    addpoints(glove_quat_an_x, t(i), glove_raw_cp_quat(i,2));
    addpoints(glove_quat_an_y, t(i), glove_raw_cp_quat(i,3));
    addpoints(glove_quat_an_z, t(i), glove_raw_cp_quat(i,4));

    addpoints(psm2_trans_an_x, t(i), psm2_cp_tran_rescaled_align(1,i));
    addpoints(psm2_trans_an_y, t(i), psm2_cp_tran_rescaled_align(2,i));
    addpoints(psm2_trans_an_z, t(i), psm2_cp_tran_rescaled_align(3,i));

    addpoints(psm2_quat_an_w, t(i), psm2_cp_quat_align(i,1));
    addpoints(psm2_quat_an_x, t(i), psm2_cp_quat_align(i,2));
    addpoints(psm2_quat_an_y, t(i), psm2_cp_quat_align(i,3));
    addpoints(psm2_quat_an_z, t(i), psm2_cp_quat_align(i,4));
    
    t_end_it = toc(t_start_it);
    pause(t_diff(i) - t_end_it);
end
t_end_overall = toc(t_start_overall);
fprintf("Elapsed time: %f (s)\n", t_end_overall);

% Plot the ending point
hold(ax_list, 'off');

%%
clc;
close all;