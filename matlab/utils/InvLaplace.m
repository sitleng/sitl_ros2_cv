function exp_At = InvLaplace(A)
syms s t
exp_At(t) = ilaplace(inv(s*eye(2)-A));
end