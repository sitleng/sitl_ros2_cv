%% Load dataset
clear;clc;

addpath(pathdef);

bag = rosbag('/home/leo/aruco_data/ecm_calib_data_fix_rot.bag');
% bag = rosbag('/home/leo/aruco_data/ecm_calib_data_new.bag');
ecm_aruco_bag = select(bag,'Topic','/ECM/ARUCO/setpoint_cp');
ecm_joint_bag = select(bag,'Topic','/ECM/setpoint_js');
ecm_aruco_msg = readMessages(ecm_aruco_bag,'DataFormat','struct');
ecm_joint_msg = readMessages(ecm_joint_bag,'DataFormat','struct');

%% Extract and organize data
% Number of samples in each bags
temp1 = size(ecm_aruco_msg);
ecm_N = temp1(1);

% Get gst(0)
ecm_gst0 = transmsg2g(ecm_aruco_msg{1}.Transform);

% Save the joint values of each psm
ecm_joint_ths = zeros(ecm_N,4);
for i=1:ecm_N
    ecm_joint_ths(i,:) = ecm_joint_msg{i}.Position';
end

% Save the g(th)s of ecm
ecm_gths = zeros(4,4,ecm_N);
for i=1:ecm_N
    ecm_gths(:,:,i) = transmsg2g(ecm_aruco_msg{i}.Transform);
end

% Save the g(th)g(0)^(-1) matrices
ecm_trans = zeros(4,4,ecm_N);
for i=1:ecm_N
    ecm_trans(:,:,i) = ecm_gths(:,:,i)/ecm_gst0;
end

% Save translation, quat of the tfs
ecm_transquat = zeros(7,ecm_N);
for i=1:ecm_N
    g = ecm_trans(:,:,i);
    [trans,quat] = g2transquat(g);
    ecm_transquat(:,i) = [trans;quat];
end

%% Use fmincon to solve
rng('shuffle');

% Algorithms: interior-point, sqp, active-set, 
opts = optimoptions(@fmincon,'Display','iter','MaxFunctionEvaluations',100000, ...
    'MaxIterations',2000,'OptimalityTolerance',1e-8,'StepTolerance',1e-8, ...
    'Algorithm','interior-point','UseParallel',true);

% Algorithms: interior-point, levenberg-marquardt, trust-region-reflective
% opts = optimoptions(@lsqnonlin,'Display','iter','MaxFunctionEvaluations',100000, ...
%     'MaxIterations',2000,'OptimalityTolerance',1e-6,'StepTolerance',1e-6, ...
%     'Algorithm','interior-point','UseParallel',true);
% opts = optimoptions(@lsqcurvefit,'Display','iter','MaxFunctionEvaluations',100000, ...
%     'MaxIterations',2000,'OptimalityTolerance',1e-6,'StepTolerance',1e-6, ...
%     'Algorithm','interior-point','UseParallel',true);

gs = GlobalSearch('Display','iter','NumTrialPoints',1000,'NumStageOnePoints',200);
% ms = MultiStart('Display','iter','StartPointsToRun','bounds-ineqs');

while true
    ecm_x0 = rand(34,1);
    ecm_x0(4:6) = ecm_x0(4:6)/norm(ecm_x0(4:6));
    ecm_x0(10:12) = ecm_x0(10:12)/norm(ecm_x0(10:12));
    ecm_x0(13:15) = ecm_x0(13:15)/norm(ecm_x0(13:15));
    ecm_x0(16:18) = zeros(3,1);
    ecm_x0(22:24) = ecm_x0(22:24)/norm(ecm_x0(22:24));
    % problem = createOptimProblem('fmincon','x0',ecm_x0,'objective', ...
    %     @(ecm_x)objfun_ecm(ecm_x,ecm_N,ecm_trans,ecm_joint_ths),'options',opts);
    problem = createOptimProblem('fmincon','x0',ecm_x0,'objective', ...
        @(ecm_x)objfun_ecm(ecm_x,ecm_N,ecm_trans,ecm_joint_ths),'nonlcon', ...
        @(ecm_x)cond_ecm(ecm_x),'options',opts);
    % problem = createOptimProblem('lsqnonlin','x0',ecm_x0,'objective', ...
    %     @(ecm_x)objfun_ecm(ecm_x,ecm_N,ecm_trans,ecm_joint_ths),'nonlcon', ...
    %     @(ecm_x)cond_ecm(ecm_x),'options',opts);
    % problem = createOptimProblem('lsqcurvefit','x0',ecm_x0,'objective', ...
    %     @(ecm_x,xdata)objfun_ecm_v2(ecm_x,xdata),'xdata',ecm_joint_ths, ...
    %     'ydata',ecm_transquat,'nonlcon',@(ecm_x)cond_ecm(ecm_x),'options',opts);
    [ecm_x,ecm_fval,exitflag,~] = run(gs,problem);
    % [ecm_x,ecm_fval,exitflag,~] = run(ms,problem,100);
    if exitflag > 0
        break;
    end
end


%% Save the solution
save('/home/leo/aruco_data/ecm_calib_results.mat')
