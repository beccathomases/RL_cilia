function replayRollout(fig, env, states, rewards)
for k = 1:length(states)
    figure(fig);
    clf(fig);
    env.render(states(k));
    drawnow;
    % reattach button after clf/render
    makeReplayButton(fig, env, states, rewards);
    if k == 1
        title('Greedy Rollout Step 0');
    else
        title(sprintf('Greedy Rollout Step %d, r = %.4g', k-1, rewards(k-1)));
    end
    drawnow;
    pause(0.1);
end
end