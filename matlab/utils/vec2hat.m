function out = vec2hat(in)
[r, c] = size(in);
if r~=3 || c~=1
    error("Incorrect vector size, it should be 3x1")
end
out = [0     -in(3)  in(2);
       in(3)   0    -in(1);
      -in(2)  in(1)    0  ];
end

