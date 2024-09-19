function g = tfstamped2g(msg)
tvec = zeros(3,1);
quat = zeros(1,4);
tvec(1) = msg.Transform.Translation.X;
tvec(2) = msg.Transform.Translation.Y;
tvec(3) = msg.Transform.Translation.Z;
quat(1) = msg.Transform.Rotation.W;
quat(2) = msg.Transform.Rotation.X;
quat(3) = msg.Transform.Rotation.Y;
quat(4) = msg.Transform.Rotation.Z;
R = quat2rotm(quat);
g = [R tvec;0 0 0 1];
end