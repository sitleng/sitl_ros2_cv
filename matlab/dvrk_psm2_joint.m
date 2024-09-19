%% Load dataset
clear;clc;

addpath(pathdef);

bag = rosbag('/home/leo/aruco_data/psm2_calib_data_fix_rot.bag');
psm2_aruco_bag = select(bag,'Topic','/PSM2/ARUCO/setpoint_cp');
psm2_joint_bag = select(bag,'Topic','/PSM2/setpoint_js');
psm2_aruco_msg = readMessages(psm2_aruco_bag,'DataFormat','struct');
psm2_joint_msg = readMessages(psm2_joint_bag,'DataFormat','struct');

%% Extract and organize data
% Number of samples in each bags
temp1 = size(psm2_aruco_msg);
psm2_N = temp1(1);

% Find each gst(0)
psm2_gst0 = transmsg2g(psm2_aruco_msg{1}.Transform);

% Save the joint values of each psm
psm2_joint_ths = zeros(psm2_N,6);
for i=1:psm2_N
    psm2_joint_ths(i,:) = psm2_joint_msg{i}.Position';
end

% Save the g(th)s for each psm
psm2_gths = zeros(4,4,psm2_N);
for i=1:psm2_N
    psm2_gths(:,:,i) = transmsg2g(psm2_aruco_msg{i}.Transform);
end

% Save the g(th)g(0)^(-1) matrices
psm2_trans = zeros(4,4,psm2_N);
for i=1:psm2_N
    psm2_trans(:,:,i) = psm2_gths(:,:,i)/psm2_gst0;
end

%% Use fmincon to solve
rng('shuffle');

opts = optimoptions(@fmincon,'Display','iter','MaxFunctionEvaluations',100000, ...
    'MaxIterations',2000,'OptimalityTolerance',1e-8,'StepTolerance',1e-8, ...
    'Algorithm','interior-point','UseParallel',true);

gs = GlobalSearch('Display','iter','NumTrialPoints',1000,'NumStageOnePoints',200);

while true
    psm2_x0 = rand(48,1);
    psm2_x0(4:6) = psm2_x0(4:6)/norm(psm2_x0(4:6));
    psm2_x0(10:12) = psm2_x0(10:12)/norm(psm2_x0(10:12));
    psm2_x0(16:18) = zeros(3,1);
    psm2_x0(22:24) = psm2_x0(22:24)/norm(psm2_x0(22:24));
    psm2_x0(28:30) = psm2_x0(28:30)/norm(psm2_x0(28:30));
    psm2_x0(34:36) = psm2_x0(34:36)/norm(psm2_x0(34:36));
    problem = createOptimProblem('fmincon','x0',psm2_x0,'objective', ...
        @(psm2_x)objfun_psm(psm2_x,psm2_N,psm2_trans,psm2_joint_ths),'nonlcon', ...
        @(psm2_x)cond_psm(psm2_x),'options',opts);
    [psm2_x,psm2_fval,exitflag,~] = run(gs,problem);
    if exitflag > 0
        break;
    end
end

%% Save the solution
save('/home/leo/aruco_data/psm2_calib_results.mat')
