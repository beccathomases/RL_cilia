function cilia_plot_coeff_trajectory(trajC)
    figure;
    plot3(trajC(1,:), trajC(2,:), trajC(3,:), '-o', 'LineWidth', 2);
    grid on;
    xlabel('c0'); ylabel('c1'); zlabel('c2');
    title('Greedy rollout trajectory in coefficient space');
end
