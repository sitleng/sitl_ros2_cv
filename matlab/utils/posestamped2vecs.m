function [tvec,quat] = posestamped2vecs(msg)
tvec = zeros(3,1);
quat = zeros(1,4);
tvec(1) = msg.Pose.Position.X;
tvec(2) = msg.Pose.Position.Y;
tvec(3) = msg.Pose.Position.Z;
quat(1) = msg.Pose.Orientation.W;
quat(2) = msg.Pose.Orientation.X;
quat(3) = msg.Pose.Orientation.Y;
quat(4) = msg.Pose.Orientation.Z;
end

