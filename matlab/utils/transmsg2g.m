function g = transmsg2g(msg)
tvec = zeros(3,1);
quat = zeros(1,4);
tvec(1) = msg.Translation.X;
tvec(2) = msg.Translation.Y;
tvec(3) = msg.Translation.Z;
quat(1) = msg.Rotation.W;
quat(2) = msg.Rotation.X;
quat(3) = msg.Rotation.Y;
quat(4) = msg.Rotation.Z;
R = quat2rotm(quat);
g = [R tvec;0 0 0 1];
end