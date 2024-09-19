function orig_x = inv_scale(new_x, min_o, max_o, min_n, max_n)
    orig_x = (new_x-min_n)*(max_o-min_o)/(max_n-min_n) + min_o;
end
