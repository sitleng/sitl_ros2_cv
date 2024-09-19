function ax = plot_tile(x, y, color, linewidth, box_start, box_end, wo_clutch)
    ax = nexttile;
    plot(ax, x, y, "Color", color, 'LineWidth', linewidth);
    hold on;
    if ~isempty(box_start)
        xr = xregion(x(box_start), x(box_end), 'FaceColor', '#FF00FF');
        for i = 1:numel(xr)
            xr(i).FaceAlpha = 0.1;
        end
    end
    xticks(ax, round(x(1),0):10:round(x(end),0));
    hold off;
    y_wo_clutch = y(wo_clutch);
    min_y = min(y_wo_clutch);
    max_y = max(y_wo_clutch);
    ytick_step = round((max_y - min_y)/2.2, 3);
    center = (max_y + min_y)/2;
    half_range = (max_y - min_y)/2 + (max_y - min_y)/8;
    yticks(ax, round(min_y,2):ytick_step:round(max_y,2));
    ylim(ax, [center-half_range, center+half_range]);
    xlim(ax, [x(1), x(end)+0.1]);
    ax.FontSize = 25;
end
