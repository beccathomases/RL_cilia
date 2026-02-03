function cilia_plot_value_slice(Q, cmin, cmax, fixed_c2)
%CILIA_PLOT_VALUE_SLICE Heatmap of V(s)=max_a Q(s,a) for fixed c2, varying c0,c1.
% Also overlays greedy action arrows.

    vals = cmin:cmax;
    nVals = numel(vals);

    V = zeros(nVals, nVals);     % rows for c1, cols for c0 (for plotting)
    Astar = zeros(nVals, nVals); % greedy action index

    for i1 = 1:nVals
        for i0 = 1:nVals
            c0 = vals(i0);
            c1 = vals(i1);
            c2 = fixed_c2;

            s = cilia_state_index([c0; c1; c2], cmin, cmax);
            [V(i1,i0), Astar(i1,i0)] = max(Q(s,:));
        end
    end

    figure;
    imagesc(vals, vals, V);  % x-axis=c0, y-axis=c1
    set(gca,'YDir','normal');
    axis equal tight;
    colorbar;
    xlabel('c0'); ylabel('c1');
    title(sprintf('Value slice: V(c0,c1 | c2=%d) = max_a Q', fixed_c2));
    hold on;

    % Overlay arrows for greedy action
    % Actions: 1 c0+; 2 c0-; 3 c1+; 4 c1-; 5 c2+; 6 c2-
    for i1 = 1:nVals
        for i0 = 1:nVals
            a = Astar(i1,i0);
            [u,v] = action_arrow(a);
            quiver(vals(i0), vals(i1), u, v, 0.35, 'k', 'LineWidth', 1);
        end
    end

    grid on;
    hold off;
end

function [u,v] = action_arrow(a)
    % Arrow directions in (c0,c1) plane.
    % If action modifies c2, draw a small dot-like arrow (0,0).
    switch a
        case 1, u = +0.8; v =  0;   % c0+
        case 2, u = -0.8; v =  0;   % c0-
        case 3, u =  0;   v = +0.8; % c1+
        case 4, u =  0;   v = -0.8; % c1-
        otherwise
            u = 0; v = 0;          % c2 actions not visible in this slice
    end
end
