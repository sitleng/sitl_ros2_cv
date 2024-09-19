function Adj_g = adjoint(g)
R = g(1:3,1:3);
p = g(1:3,4);
Adj_g = [R vec2hat(p)*R;zeros(3) R];
end

