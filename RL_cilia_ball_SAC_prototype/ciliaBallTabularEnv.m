function env = ciliaBallTabularEnv(P, opts)
% ciliaBallTabularEnv  Tabular RL wrapper for the 2-ball (or N-ball) cilia model.
%
% States: discrete joint angles on a grid over [-phimax(k), phimax(k)]
% Actions: all combinations of {-1,0,1} increments for each hinge
% Reward: calls your cilia_ball_reward(phi, action, P) (flux in x-direction)
% Transition: grid step + boundary handling (clip)
%
% Usage:
%   P   = setdefaultparams_ciliaball;
%   env = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));
%   s   = env.reset();
%   [s2, r] = env.step(s, 1);
%   env.render(s2);

  if nargin < 1 || isempty(P)
    if exist('setdefaultparams_ciliaball','file') == 2
      P = setdefaultparams_ciliaball;
    else
      error('Provide P, or put setdefaultparams_ciliaball.m on the path.');
    end
  end
  if nargin < 2, opts = struct(); end

  % ---- options ----
  opts = setDefault(opts, 'reset_mode', 'fixed');   % 'fixed' or 'random'
  opts = setDefault(opts, 'phi0', [0, -P.phimax(2)]); % used if reset_mode='fixed'
  opts = setDefault(opts, 'precompute', false);     % precompute R(s,a), sNext(s,a)
  opts = setDefault(opts, 'boundary', 'clip');      % only 'clip' implemented

  % sanity: you need your physics reward function on path
  if exist('cilia_ball_reward','file') ~= 2
    error('Need cilia_ball_reward.m on the MATLAB path.');
  end

  % ---- state grid ----
  Nstates = P.Nstates(:)';           % e.g. [11 21] for 2-ball
  Nh = numel(Nstates);               % number of hinges/angles
  if Nh ~= P.N
    % P.N is #balls; hinges/angles count in your code matches params.N
    % If you ever separate these, adjust here.
  end

  phiVals = cell(1,Nh);
  for k = 1:Nh
    phiVals{k} = linspace(-P.phimax(k), P.phimax(k), Nstates(k));
  end
  nS = prod(Nstates);

  % ---- action list ----
  % All combinations of {-1,0,1}^Nh, as rows
  base = [-1 0 1];
  A = allTuples(base, Nh);  % size nA x Nh
  nA = size(A,1);

  % ---- pack env ----
  env.P         = P;
  env.opts      = opts;
  env.Nstates   = Nstates;
  env.nStates   = nS;
  env.actions   = A;
  env.nActions  = nA;

  env.state2sub = @(s) ind2subv(Nstates, s);
  env.sub2state = @(sub) sub2indv(Nstates, sub);

  env.sub2phi   = @(sub) sub2phi(sub, phiVals);
  env.phi2sub   = @(phi) phi2sub(phi, phiVals);

  env.reset     = @() resetFn();
  env.step      = @(s,a) stepFn(s,a);
  env.render    = @(s) renderFn(s);

  env.Rtable    = [];
  env.Ntable    = [];

  if opts.precompute
    [env.Rtable, env.Ntable] = precomputeTables(env);
  end

  % ================= nested functions =================

  function s0 = resetFn()
    switch lower(opts.reset_mode)
      case 'fixed'
        sub0 = env.phi2sub(opts.phi0);
        s0   = env.sub2state(sub0);
      case 'random'
        sub0 = arrayfun(@(n) randi(n), Nstates);
        s0   = env.sub2state(sub0);
      otherwise
        error('opts.reset_mode must be ''fixed'' or ''random''.');
    end
  end

  function [sNext, r, isTerminal, info] = stepFn(s, aIdx)
    if aIdx < 1 || aIdx > nA, error('Action index out of range.'); end
    isTerminal = false;

    % If precomputed, just look up
    if ~isempty(env.Rtable)
      r     = env.Rtable(s, aIdx);
      sNext = env.Ntable(s, aIdx);
      info  = struct();
      return
    end

    sub  = env.state2sub(s);
    phi  = env.sub2phi(sub);
    aReq = env.actions(aIdx,:);

    % boundary handling: compute "effective" action in index space
    % (so if you try to step out of bounds, you actually stay on the boundary)
    subNext = sub + aReq;
    if strcmpi(opts.boundary,'clip')
      for k = 1:Nh
        subNext(k) = min(max(subNext(k), 1), Nstates(k));
      end
    else
      error('Only opts.boundary=''clip'' is implemented.');
    end
    aEff = subNext - sub;   % each entry in {-1,0,1}, but possibly reduced by clipping

    % reward from your physics function (uses midpoint rule internally)
    % Note: your cilia_ball_reward returns a "next_state" in continuous angles,
    % but we ignore that and use our discrete clipped transition.
    [r, ~] = cilia_ball_reward(phi, aEff, P);

    sNext = env.sub2state(subNext);

    info = struct();
    info.phi     = phi;
    info.action  = aEff;
    info.phiNext = env.sub2phi(subNext);
  end

  function renderFn(s)
    sub = env.state2sub(s);
    phi = env.sub2phi(sub);

    if exist('position_from_angle','file') ~= 2
      % fallback: just print
      fprintf('state %d, phi = [%g %g]\n', s, phi(1), phi(2));
      return
    end

    X  = position_from_angle(phi, P);
    XX = [P.X0; X];

    figure(gcf); clf;
    plot(XX(:,1), XX(:,3), 'k-', 'LineWidth', 3); hold on;
    plot(X(:,1),  X(:,3),  'r.', 'MarkerSize', 30);
    plot([-1 1],[0 0],'k','LineWidth',5);
    xlim([-1 1]); ylim([-0.25 1.5]);
    axis equal;
    title(sprintf('phi = [%0.3f, %0.3f]', phi(1), phi(2)));
    drawnow;
  end

end

% ================= helper functions (subfunctions) =================

function opts = setDefault(opts, name, val)
  if ~isfield(opts, name) || isempty(opts.(name))
    opts.(name) = val;
  end
end

function T = allTuples(vals, d)
% allTuples([-1 0 1], d) -> 3^d by d matrix of all combinations
  grids = cell(1,d);
  [grids{:}] = ndgrid(vals);
  T = zeros(numel(grids{1}), d);
  for k = 1:d
    T(:,k) = grids{k}(:);
  end
end

function sub = ind2subv(sz, idx)
% vectorized ind2sub -> row vector of subscripts
  c = cell(1, numel(sz));
  [c{:}] = ind2sub(sz, idx);
  sub = cellfun(@(x) x, c);
end

function idx = sub2indv(sz, sub)
% row vector sub -> linear index
  c = num2cell(sub);
  idx = sub2ind(sz, c{:});
end

function phi = sub2phi(sub, phiVals)
  Nh  = numel(sub);
  phi = zeros(1,Nh);
  for k = 1:Nh
    phi(k) = phiVals{k}(sub(k));
  end
end

function sub = phi2sub(phi, phiVals)
% map continuous phi to nearest discrete grid index
  Nh  = numel(phi);
  sub = zeros(1,Nh);
  for k = 1:Nh
    [~, sub(k)] = min(abs(phiVals{k} - phi(k)));
  end
end

function [R, N] = precomputeTables(env)
% Precompute reward and next-state tables for fast tabular Q-learning.
  nS = env.nStates; nA = env.nActions;
  R  = zeros(nS, nA);
  N  = zeros(nS, nA);

  for s = 1:nS
    for a = 1:nA
      [s2, r] = env.step(s, a);  % uses non-precomputed path
      R(s,a) = r;
      N(s,a) = s2;
    end
  end
end