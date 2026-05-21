% This code loads a saved tiny-NN dphi cycle and makes a movie of that cycle.
% Change the parameters below to load the files you want to look at.

clear; clc;

nreps = 10;          % how many times to repeat the cycle in the movie
pausetime = 0.01;   % pause between frames when displaying

saveMovie = true;   % true = save movie file, false = just display
fps = 10;           % frames per second for saved movie

nseeds       = 4;
nEpisodes    = 25000;
alpha0       = 0.01;
epsilon0     = 0.3;
gamma        = 0.99;
m            = 16;
stuckPenalty = 0.05;

for seeds = nseeds
    fin = sprintf(['cycle_tinyNN_dphi_seed%d_g%1.3f_eps0%1.2f_' ...
                   'alp0%1.3f_nEpisode%d_m%d_sp%1.2f.mat'], ...
                   seeds, gamma, epsilon0, alpha0, nEpisodes, m, stuckPenalty);

    fout = sprintf(['cycle_tinyNN_dphi_seed%d_g%1.3f_eps0%1.2f_' ...
                    'alp0%1.3f_nEpisode%d_m%d_sp%1.2f.mp4'], ...
                    seeds, gamma, epsilon0, alpha0, nEpisodes, m, stuckPenalty);

    if ~isfile(fin)
        fprintf('Could not find %s -- skipping.\n', fin);
        continue
    end

    S = load(fin);

    if ~isfield(S, 'cycle_states')
        fprintf('File %s does not contain cycle_states -- skipping.\n', fin);
        continue
    end

    cycle_states = S.cycle_states;

    P = setdefaultparams_ciliaball;
    env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

    playCycleMovie(env, cycle_states, nreps, pausetime, saveMovie, fout, fps)
end


% ================= helper function below =================
function playCycleMovie(env, cycle_states, nCyclesToShow, pauseTime, saveMovie, movieFile, fps)
% playCycleMovie  Animate a detected cycle for the cilia-ball model.
%
% Inputs:
%   env            environment struct from ciliaBallTabularEnv
%   cycle_states   vector of state indices in cycle order
%                  (can include repeated closing state at end)
%   nCyclesToShow  number of times to repeat the cycle
%   pauseTime      pause between frames when displaying
%   saveMovie      true/false, whether to save a video
%   movieFile      output filename, e.g. 'mymovie.mp4'
%   fps            frames per second for saved video

    if nargin < 3 || isempty(nCyclesToShow)
        nCyclesToShow = 10;
    end
    if nargin < 4 || isempty(pauseTime)
        pauseTime = 0.1;
    end
    if nargin < 5 || isempty(saveMovie)
        saveMovie = false;
    end
    if nargin < 6 || isempty(movieFile)
        movieFile = 'cycle_movie.mp4';
    end
    if nargin < 7 || isempty(fps)
        fps = 10;
    end

    % If cycle stored as [... s0] with repeated closing state, remove last state
    if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
        cycle_core = cycle_states(1:end-1);
    else
        cycle_core = cycle_states;
    end

    if isempty(cycle_core)
        error('cycle_states is empty.');
    end

    nPhase = length(cycle_core);

    if saveMovie
        v = VideoWriter(movieFile, 'MPEG-4');
        v.FrameRate = fps;
        open(v);
    end

    figure;
    for k = 1:nCyclesToShow
        for j = 1:nPhase
            s = cycle_core(j);

            % Convert state -> subscripts -> hinge angles
            sub = env.state2sub(s);
            phi = env.sub2phi(sub);

            % Compute positions
            X  = position_from_angle(phi, env.P);
            XX = [env.P.X0; X];

            % Draw
            clf;
            plot(XX(:,1), XX(:,3), 'k-', 'LineWidth', 3); hold on;
            plot(X(:,1),  X(:,3),  'r.', 'MarkerSize', 30);
            plot([-1 1], [0 0], 'k', 'LineWidth', 5);

            xlim([-1 1]);
            ylim([-0.25 1.5]);
            axis equal;
            grid on;

            title(sprintf('Phase %02d/%02d, Period %02d/%02d', ...
                j, nPhase, k, nCyclesToShow));

            drawnow;

            if saveMovie
                frame = getframe(gcf);
                writeVideo(v, frame);
            end

            if pauseTime > 0
                pause(pauseTime);
            end
        end
    end

    if saveMovie
        close(v);
        fprintf('Saved movie to %s\n', movieFile);
    end
end