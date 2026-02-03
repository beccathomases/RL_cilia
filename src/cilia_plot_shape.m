function cilia_plot_shape(out)
%CILIA_PLOT_SHAPE Plot the centerline and the wall y=0.

    plot(out.x, out.y, 'LineWidth', 2); hold on;
    yline(0, '--');
    plot(out.x(1), out.y(1), 'ko', 'MarkerFaceColor','k');   % base
    plot(out.x(end), out.y(end), 'ro', 'MarkerFaceColor','r'); % tip
    axis equal; grid on;
    xlabel('x'); ylabel('y');
    legend('centerline','wall y=0','base','tip','Location','best');
    hold off;
end
