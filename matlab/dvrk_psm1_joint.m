%% Load dataset
clear;clc;

addpath(pathdef);

bag = rosbag('/home/leo/aruco_data/psm1_calib_data_fix_rot.bag');
psm1_aruco_bag = select(bag,'Topic','/PSM1/ARUCO/setpoint_cp');
psm1_joint_bag = select(bag,'Topic','/PSM1/setpoint_js');
psm1_aruco_msg = readMessages(psm1_aruco_bag,'DataFormat','struct');
psm1_joint_msg = readMessages(psm1_joint_bag,'DataFormat','struct');

%% Extract and organize data
% Number of samples in each bags
temp1 = size(psm1_aruco_msg);
psm1_N = temp1(1);

% Find each gst(0)
psm1_gst0 = transmsg2g(psm1_aruco_msg{1}.Transform);

% Save the joint values of each psm
psm1_joint_ths = zeros(psm1_N,6);
for i=1:psm1_N
    psm1_joint_ths(i,:) = psm1_joint_msg{i}.Position';
end

% Save the g(th)s for each psm
psm1_gths = zeros(4,4,psm1_N);
for i=1:psm1_N
    psm1_gths(:,:,i) = transmsg2g(psm1_aruco_msg{i}.Transform);
end

% Save the g(th)g(0)^(-1) matrices
psm1_trans = zeros(4,4,psm1_N);
for i=1:psm1_N
    psm1_trans(:,:,i) = psm1_gths(:,:,i)/psm1_gst0;
end


%% Use fmincon to solve
rng('shuffle');

opts = optimoptions(@fmincon,'Display','iter','MaxFunctionEvaluations',100000, ...
    'MaxIterations',2000,'OptimalityTolerance',1e-6,'StepTolerance',1e-6, ...
    'Algorithm','interior-point','UseParallel',true);

gs = GlobalSearch('Display','iter','NumTrialPoints',1000,'NumStageOnePoints',200);

while true
    psm1_x0 = rand(48,1);
    psm1_x0(4:6) = psm1_x0(4:6)/norm(psm1_x0(4:6));
    psm1_x0(10:12) = psm1_x0(10:12)/norm(psm1_x0(10:12));
    psm1_x0(16:18) = zeros(3,1);
    psm1_x0(22:24) = psm1_x0(22:24)/norm(psm1_x0(22:24));
    psm1_x0(28:30) = psm1_x0(28:30)/norm(psm1_x0(28:30));
    psm1_x0(34:36) = psm1_x0(34:36)/norm(psm1_x0(34:36));
    problem = createOptimProblem('fmincon','x0',psm1_x0,'objective', ...
        @(psm1_x)objfun_psm(psm1_x,psm1_N,psm1_trans,psm1_joint_ths),'nonlcon', ...
        @(psm1_x)cond_psm(psm1_x),'options',opts);
    [psm1_x,psm1_fval,exitflag,~] = run(gs,problem);
    if exitflag > 0
        break;
    end
end


%% Save the solution
save('/home/leo/aruco_data/psm1_calib_results.mat')
