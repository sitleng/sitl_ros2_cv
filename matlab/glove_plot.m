%%
clear; clc;

%%

list_p_dist = [];
list_R_dist = [];
list_q_dist = [];

%% Load dataset

addpath(pathdef);

user = 'arman';
trial = 5;
% is_on_off = false;
% filename = sprintf('C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\icra_2025\\%s_%d.bag', user, trial);


is_on_off = true;
filename = 'C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\icra_2025\\plot_test_2.bag';

% is_on_off = false;
% filename = 'C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\icra_2025\\test_gesture_delay.bag';

bag = rosbag(filename);

disp('Available Topics:');
disp(bag.AvailableTopics);

% glove_tf_bag   = select(bag,'Topic','/vive_tracker_transform');
glove_tf_bag     = select(bag,'Topic','/tracker_current_pos_tf');
glove_raw_tf_bag = select(bag,'Topic','/tracker_current_raw_data');
psm2_local_bag   = select(bag,'Topic','/PSM2/custom/local/setpoint_cp');
% mtml_local_bag   = select(bag,'Topic','/MTML/local/setpoint_cp');
clutch_bag       = select(bag,'Topic','/glove/left/fist');

% glove_jaw_bag    = select(bag,'Topic','/glove/left/angle');
glove_jaw_bag    = select(bag,'Topic','/glove/left/raw_angle');

psm2_jaw_bag     = select(bag,'Topic','/PSM2/jaw/setpoint_js');


glove_tf_msg     = readMessages(glove_tf_bag, 'DataFormat','struct');
glove_raw_tf_msg = readMessages(glove_raw_tf_bag, 'DataFormat','struct');
psm2_local_msg   = readMessages(psm2_local_bag,'DataFormat','struct');
% mtml_local_msg   = readMessages(mtml_local_bag,'DataFormat','struct');
clutch_msg       = readMessages(clutch_bag,'DataFormat','struct');
glove_jaw_msg    = readMessages(glove_jaw_bag,'DataFormat','struct');
psm2_jaw_msg     = readMessages(psm2_jaw_bag,'DataFormat','struct');

if is_on_off
    on_off_bag       = select(bag,'Topic','/glove/left/on_off');
    on_off_msg       = readMessages(on_off_bag,'DataFormat','struct');
    clear on_off_bag;
end

clear bag glove_tf_bag glove_raw_tf_bag psm2_local_bag ... % mtml_local_bag ...
    clutch_bag glove_jaw_bag psm2_jaw_bag;

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
% mtml_local_cp = zeros(4, 4, full_N);
on_off_flag   = zeros(1, full_N);

% glove_psm2_offset = vecs2g([0; 0; 1]*pi, [0; 0; 0])*...
%     vecs2g([0; 1; 0]*(-pi/2), [0; 0; 0]);

for i = 1:full_N
    t(i)                 = ext_ros_stamp(glove_raw_tf_msg{i});
    glove_cp(:,:,i)      = transmsg2g(glove_tf_msg{i}.Transform);
    glove_raw_cp(:,:,i)  = transmsg2g(glove_raw_tf_msg{i}.Transform);
    psm2_local_cp(:,:,i) = posestamped2g(psm2_local_msg{i});
    % mtml_local_cp(:,:,i) = posestamped2g(mtml_local_msg{i});
    clutch_flag(i)       = clutch_msg{i}.Data;
    glove_jaw(i)         = glove_jaw_msg{i}.Data;
    psm2_jaw(i)          = psm2_jaw_msg{i}.Position;
    if is_on_off
        on_off_flag(i)   = on_off_msg{i}.Data;
    end
end

t = sort(t - t(1));

clutch_start = find(diff(clutch_flag)>0);
clutch_end   = find(diff(clutch_flag)<0);

% Save a copy of the full version for PSM2
psm2_local_cp_full = psm2_local_cp;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(user,'diego') && trial == 1
    temp1 = find(t>315);
elseif strcmp(user,'paula') && trial == 4
    temp1 = 1:clutch_start(end);
    clutch_start = clutch_start(1:end-1);
    clutch_end   = clutch_end(1:end-1);
elseif is_on_off
    temp1 = find(diff(on_off_flag)>0):find(diff(on_off_flag)<0);
else
    temp1 = 1:full_N - 50;
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

% N = clutch_end(end) + 50;

t             = t(temp1);
glove_cp      = glove_cp     (:,:,temp1);
glove_raw_cp  = glove_raw_cp (:,:,temp1);
psm2_local_cp = psm2_local_cp(:,:,temp1);
% mtml_local_cp = mtml_local_cp(:,:,temp1);
clutch_flag   = clutch_flag  (temp1);
glove_jaw     = glove_jaw    (temp1);
psm2_jaw      = psm2_jaw     (temp1);

if is_on_off
    on_off_start  = 1;
    on_off_end    = find(abs(t - (t(1) + 3)) <= 1e-1, 1);
    clear on_off_msg;
end

clear glove_tf_msg glove_raw_tf_msg psm2_local_msg ... % mtml_local_msg ...
    clutch_msg glove_jaw_msg psm2_jaw_msg on_off_msg;

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

eps = 0.1;
glove_cp_quat     = clean_quat(glove_cp_quat, eps);
glove_raw_cp_quat = clean_quat(glove_raw_cp_quat, eps);
psm2_cp_quat      = clean_quat(psm2_cp_quat, eps);
psm2_cp_quat_full = clean_quat(psm2_cp_quat_full, eps);

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

plot_tile(t, glove_raw_cp_tran(3,:), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{z}$$', 'FontSize',40, 'Interpreter','latex');

plot_tile(t, psm2_cp_tran(3,:), 'b', 1.5, clutch_start, clutch_end, wo_clutch);
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

plot_tile(t, glove_raw_cp_quat(:,1), '#EDB120', 1.5, clutch_start, clutch_end, wo_clutch);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');

plot_tile(t, psm2_cp_quat(:,1), '#EDB120', 1.5, clutch_start, clutch_end, wo_clutch);

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

%% Plot the trajectory for each axis (Translation, On Off)

figure(1);

tl = tiledlayout(3,2);

ax = plot_tile_on_off(t, glove_raw_cp_tran(1,:), 'r', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
xticklabels(ax,{});
title(ax, '$$\bf{Glove}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{x \ (m)}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile_on_off(t, psm2_cp_tran(1,:), 'r', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
xticklabels(ax,{});
title(ax, '$$\bf{PSM2}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile_on_off(t, glove_raw_cp_tran(2,:), 'g', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{y}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile_on_off(t, psm2_cp_tran(2,:), 'g', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
xticklabels(ax,{});

plot_tile_on_off(t, glove_raw_cp_tran(3,:), 'b', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{z}$$', 'FontSize',40, 'Interpreter','latex');

plot_tile_on_off(t, psm2_cp_tran(3,:), 'b', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');

ylabel(tl, '$$\bf{Translation [x, y, z] (m)}$$', 'FontSize', 40, 'Interpreter','latex');
xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize',40, 'Interpreter','latex');

tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%% Save Plot (Translation, On Off)
exportgraphics(gcf, "C:\\Users\\kiwil\\Box\\SITL\\ICRA2025\\glove_psm2_trans.pdf")

%% Plot the trajectory for each axis (Rotation, On Off)

figure(2);

tl = tiledlayout(4,2);

ax = plot_tile_on_off(t, glove_raw_cp_quat(:,2), 'r', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
xticklabels(ax,{});
title(ax, '$$\bf{Glove}$$', 'FontSize',40, 'Interpreter','latex');
% ylabel(ax, '$$\bf{x}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile_on_off(t, psm2_cp_quat(:,2), 'r', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
xticklabels(ax,{});
title(ax, '$$\bf{PSM2}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile_on_off(t, glove_raw_cp_quat(:,3), 'g', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{y}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile_on_off(t, psm2_cp_quat(:,3), 'g', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
xticklabels(ax,{});

ax = plot_tile_on_off(t, glove_raw_cp_quat(:,4), 'b', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
xticklabels(ax,{});
% ylabel(ax, '$$\bf{z}$$', 'FontSize',40, 'Interpreter','latex');

ax = plot_tile_on_off(t, psm2_cp_quat(:,4), 'b', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
xticklabels(ax,{});

plot_tile_on_off(t, glove_raw_cp_quat(:,1), '#EDB120', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
% xlabel(ax, '$$\bf{t}$$', 'FontSize',40, 'Interpreter','latex');

plot_tile_on_off(t, psm2_cp_quat(:,1), '#EDB120', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);

ylabel(tl, '$$\bf{Quaternion [x, y, z, w] (rad)}$$', 'FontSize',40, 'Interpreter','latex');
xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize', 40, 'Interpreter','latex');

tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%% Save Plot (Rotation, On Off)
exportgraphics(gcf, "C:\\Users\\kiwil\\Box\\SITL\\ICRA2025\\glove_psm2_rot.pdf")

%% Plot the Glove-Jaw Angles (On Off)

figure(4);

tl = tiledlayout(1,2);

ax = plot_tile_on_off(t, glove_jaw, '#A2142F', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
title(ax, '$$\bf{Glove}$$', 'FontSize', 40, 'Interpreter','latex');
ylabel(ax, '$$\bf{Distance (m)}$$', 'FontSize', 40, 'Interpreter','latex');

ax = plot_tile_on_off(t, psm2_jaw, '#A2142F', 1.5, clutch_start, clutch_end, wo_clutch, on_off_start, on_off_end);
title(ax, '$$\bf{PSM2}$$', 'FontSize', 40, 'Interpreter','latex');
ylabel(ax, '$$\bf{Jaw Angle (rad)}$$', 'FontSize', 40, 'Interpreter','latex');

xlabel(tl, '$$\bf{t \ (s)}$$', 'FontSize', 40, 'Interpreter','latex');

tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%% Save Plot (Rotation, On Off)
exportgraphics(gcf, "C:\\Users\\kiwil\\Box\\SITL\\ICRA2025\\glove_psm2_jaw.pdf")

%% Load Min_Max for rescaling
% filename = sprintf('C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\min_max\\min_max_%s_%d.txt', user, trial);
filename = 'C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\min_max\\min_max_plot.txt';

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

%%

cp_tran_offset = zeros(3, N);
cp_quat_offset = zeros(N, 4);
cp_quat_offset(:,1) = 1;

pt_starts = [1, clutch_end];
pt_ends   = [clutch_start, N];

for i = 1:length(pt_starts)
    tran_offset = mean(glove_cp_tran(:, pt_starts(i):pt_ends(i)) - psm2_cp_tran_rescaled(:, pt_starts(i):pt_ends(i)), 2);
    quat_offset = compact(meanrot(quaternion(quatmultiply(glove_cp_quat(pt_starts(i):pt_ends(i), :), quatinv(psm2_cp_quat(pt_starts(i):pt_ends(i), :))))));
    cp_tran_offset(:, pt_starts(i):pt_ends(i)) = repmat(tran_offset, 1, pt_ends(i)-pt_starts(i) + 1);
    cp_quat_offset(pt_starts(i):pt_ends(i), :) = repmat(quat_offset, pt_ends(i)-pt_starts(i) + 1, 1);
end

psm2_cp_tran_rescaled = psm2_cp_tran_rescaled + cp_tran_offset;
psm2_cp_quat = quatmultiply(cp_quat_offset, psm2_cp_quat);

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

% for i = 1:height(glove_cp_tran) + width(glove_cp_quat)
%     if i <=3
%         D(i) = finddelay(glove_cp_tran(i, :), psm2_cp_tran_rescaled(i, :));
%     else
%         D(i) = finddelay(glove_cp_quat(:, i-3), psm2_cp_quat(:, i-3));
%     end
% end

% for i = 1:height(glove_cp_tran) + width(glove_cp_quat)
%     if i <=3
%         max_temp = max(glove_cp_tran(i,wo_clutch));
%         min_temp = min(glove_cp_tran(i,wo_clutch));
%         [~, ~, D(i)] = alignsignals( ...
%             glove_cp_tran(i, :), psm2_cp_tran_rescaled(i, :), Method="xcorr");
%     else
%         max_temp = max(glove_cp_quat(wo_clutch, i-3));
%         min_temp = min(glove_cp_quat(wo_clutch, i-3));
%         [~, ~, D(i)] = alignsignals( ...
%             glove_cp_quat(:, i-3), psm2_cp_quat(:, i-3), Method="xcorr");
%     end
% end

% for i = 1:height(glove_cp_tran) + width(glove_cp_quat)
%     if i <=3
%         max_temp = max(glove_cp_tran(i,wo_clutch));
%         min_temp = min(glove_cp_tran(i,wo_clutch));
%         [~, ~, D(i)] = alignsignals( ...
%             glove_cp_tran(i, :), psm2_cp_tran_rescaled(i, :), Method="npeak", ...
%             PeakNum=3, MinPeakProminence=0.1*(max_temp-min_temp));
%     else
%         max_temp = max(glove_cp_quat(wo_clutch, i-3));
%         min_temp = min(glove_cp_quat(wo_clutch, i-3));
%         [~, ~, D(i)] = alignsignals( ...
%             glove_cp_quat(:, i-3), psm2_cp_quat(:, i-3), Method="npeak", ...
%             PeakNum=3, MinPeakProminence=0.1*(max_temp-min_temp));
%     end
% end

avg_D = round(mean(rmoutliers(abs(D))), 0);
% avg_D = round(mean(D(1:3)), 0);
% avg_D = 12;

psm2_cp_tran_rescaled_align = zeros(size(psm2_cp_tran_rescaled));
psm2_cp_tran_rescaled_align(:, 1:end-avg_D) = psm2_cp_tran_rescaled(:, avg_D+1:end);
psm2_cp_tran_rescaled_align(:, end-avg_D+1:end) = psm2_cp_tran_full_rescaled(:,temp1(end)+1:temp1(end)+avg_D);

psm2_cp_R_align = zeros(size(psm2_cp_R));
psm2_cp_R_align(:,:,1:end-avg_D) = psm2_cp_R(:,:,avg_D+1:end);
psm2_cp_R_align(:,:,end-avg_D+1:end) = psm2_cp_R_full(:,:,temp1(end)+1:temp1(end)+avg_D);

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

%% Plot the trajectory in 3D space (When clutched)

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
plot(ax, t_temp, glove_cp_tran(1,temp), t_temp, psm2_cp_tran_rescaled(1,temp), ...
    'LineWidth', 1.5);
legend({'Glove','PSM Rescaled'}, "Location", 'best', 'FontSize',20);
xticks(ax, round(t_temp(1),0):5:round(t_temp(end),0));
yticks(ax, round(min_y,2):ytick_step:round(max_y,2));
ylim(ax, [center-half_range, center+half_range]);
xlim(ax, [t_temp(1), t_temp(end)+0.1]);
ax.FontSize = 25;

ax = nexttile;
plot(ax, t_temp, glove_cp_tran(1,temp), t_temp, psm2_cp_tran_rescaled_align(1,temp), ...
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
