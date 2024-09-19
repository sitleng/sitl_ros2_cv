%%
clear;
clc;

%%

% Specify the path to your ROS bag file
user = 'arman';
trial = 5;
filename = sprintf('C:\\Users\\kiwil\\Box\\SITL\\glove_rec\\icra_2025\\glove_console_%s%d.bag', user, trial);

% Create a rosbag object
bag = rosbag(filename);

% Display bag information
disp('ROS Bag Information:');
disp(['File Path: ' filename]);
disp(['Start Time: ' char(bag.StartTime)]);
disp(['End Time: ' char(bag.EndTime)]);

disp(['Number of Messages: ' num2str(bag.NumMessages)]);

% List the available topics in the bag
disp('Available Topics:');
disp(bag.AvailableTopics);

%% Load dataset

mtml_bag   = select(bag,'Topic','/MTML/measured_cp');
psm2_bag   = select(bag,'Topic','/PSM2/custom/setpoint_cp');
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

%% Get the translations of the glove and the PSM2

mtml_cp_tran = reshape(mtml_cp(1:3,4,:),[3,N]);
psm2_cp_tran = reshape(psm2_cp(1:3,4,:),[3,N]);
mtml_cp_tran = mtml_cp_tran(:,wo_clutch);
psm2_cp_tran = psm2_cp_tran(:,wo_clutch);

%% Find travel distance

clc;

fprintf('Dataset: %s %d\n', user, trial);

fprintf('Duration: %.3f (s)\n', t(end) - t(1));

fprintf('Travel Distance of PSM (Console): %.3f (m)\n', travel_dist(psm2_cp_tran));

%% Animated 3D Plot

% Initialize the figure
% Get screen size
screenSize = get(0, 'ScreenSize');
screenWidth = screenSize(3);
screenHeight = screenSize(4);

% Define the position for the right half of the screen
figurePosition = [screenWidth/2, 0, screenWidth/2, screenHeight];

figure('Position', figurePosition);

% Initial view
azimuth = 2;
elevation = 30; % You can adjust this value as needed
view(azimuth, elevation);

hold on;

% Labels, legend, and grid
fontsize_label = 25;
fonsize_title  = 30;
ax = gca;
ax.FontSize = 24; 
xlabel('X-axis','FontSize', fontsize_label);
ylabel('Y-axis','FontSize', fontsize_label);
zlabel('Z-axis','FontSize', fontsize_label);
% legend('hook position','starting point','trajectory', 'ending point','FontSize', fonsize_title)
grid on;

for i = 1:full_N
    pause_time=abs(t(i)-t(i+1));
    % Animation loop for the trajectory
    scatter3(mtml_cp_tran(1,i), mtml_cp_tran(2,i), mtml_cp_tran(3,i), 50, 'filled','r');
    pause(pause_time); 
end

% Plot the ending point
hold off;