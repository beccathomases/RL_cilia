function [env, obsInfo, actInfo, P, opts] = make_cilia2ball_rltoolbox_env(userOpts)
% MAKE_CILIA2BALL_RLTOOLBOX_ENV
%
% Closure + transport SAC version.
%
% Observation:
%   [phi1; phi2; dphi1; dphi2; closureFrac; armedFlag; spanFrac; cycleFlux; cycleProgress]
%
% Reward idea:
%   - step penalty
%   + tiny raw-flux shaping
%   + gated closure-improvement shaping
%   + gated transport-improvement shaping
%   + cycle completion bonus

    if nargin < 1
        userOpts = struct();
    end

    P = setdefaultparams_ciliaball;

    % --------------------------------------------------------
    % defaults
    % --------------------------------------------------------
    opts = struct();

    % episode / reset
    opts.maxSteps = 250;
    opts.resetMode = 'random';
    opts.stepScaleFactor = 0.5;
    opts.pDemoReset = 0.7;
    opts.demoPhiList = [];
    opts.demoNoiseScale = 0.25 * P.dphi(:);

    % dense reward pieces
    opts.stepPenalty   = 0.0010;
    opts.fluxStepScale = 0.0020;

    % gated shaping
    opts.lambdaClosure   = 0.10;
    opts.lambdaTransport = 0.05;
    opts.sigmaClosureFrac = 0.15;

    % cycle event parameters
    opts.spanArmThreshold   = 0.45;
    opts.closeThreshold     = 0.18;
    opts.minCycleSteps      = 10;
    opts.fluxThreshold      = 0.02;
    opts.lambdaCycle        = 4.0;
    opts.armBonus           = 0.0;
    opts.resetAnchorOnCycle = true;

    % bounds / observation scaling
    opts.cycleFluxMax = 2.0;

    % optional stopping
    opts.terminateOnCycle = false;

    % physical index used for span
    opts.xDispPointIndex = P.N;

    % --------------------------------------------------------
    % override from userOpts
    % --------------------------------------------------------
    fn = fieldnames(userOpts);
    for k = 1:numel(fn)
        opts.(fn{k}) = userOpts.(fn{k});
    end

    if ~isfield(userOpts,'stepScale')
        opts.stepScale = opts.stepScaleFactor * P.dphi(:);
    else
        opts.stepScale = userOpts.stepScale(:);
    end

    opts.demoNoiseScale = opts.demoNoiseScale(:);
    opts.stepScale = opts.stepScale(:);

    if ~isfield(userOpts,'cycleFluxMax')
        opts.cycleFluxMax = 2.0;
    end

    if strcmpi(opts.resetMode,'mixed_demo') && isempty(opts.demoPhiList)
        error('resetMode=''mixed_demo'' requires opts.demoPhiList to be nonempty.');
    end

    % --------------------------------------------------------
    % observation specification
    % --------------------------------------------------------
    obsLow  = [ ...
        -P.phimax(:); ...
        -opts.stepScale(:); ...
        0; ...                 % closureFrac
        0; ...                 % armedFlag
        0; ...                 % spanFrac
        -opts.cycleFluxMax; ...
        0 ...                  % cycleProgress
        ];

    obsHigh = [ ...
         P.phimax(:); ...
         opts.stepScale(:); ...
         1.5; ...
         1; ...
         1; ...
         opts.cycleFluxMax; ...
         1 ...
         ];

    obsInfo = rlNumericSpec([9 1], ...
        LowerLimit = obsLow, ...
        UpperLimit = obsHigh);

    obsInfo.Name = "ciliaState";
    obsInfo.Description = "[phi1; phi2; dphi1; dphi2; closureFrac; armedFlag; spanFrac; cycleFlux; cycleProgress]";

    % --------------------------------------------------------
    % action specification
    % --------------------------------------------------------
    actInfo = rlNumericSpec([2 1], ...
        LowerLimit = -ones(2,1), ...
        UpperLimit =  ones(2,1));

    actInfo.Name = "hingeControl";

    % --------------------------------------------------------
    % environment
    % --------------------------------------------------------
    stepHandle  = @(action, loggedSignals) cilia_stepFcn(action, loggedSignals, P, opts);
    resetHandle = @() cilia_resetFcn(P, opts);

    env = rlFunctionEnv(obsInfo, actInfo, stepHandle, resetHandle);
end