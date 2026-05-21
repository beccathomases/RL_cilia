function [nextObservation, reward, isDone, loggedSignals] = cilia_stepFcn(action, loggedSignals, P, opts)
% CILIA_STEPFCN
%
% Reward:
%   reward =
%      - stepPenalty
%      + fluxStepScale * rawFlux
%      + gated positive closure improvement
%      + gated positive transport improvement
%      + armBonus on first arming event
%      + lambdaCycle * cycleFluxNew on cycle completion

    % --------------------------------------------------------
    % 1. clip action
    % --------------------------------------------------------
    action = action(:);
    action = max(-ones(2,1), min(ones(2,1), action));

    % --------------------------------------------------------
    % 2. raw flux and new shape
    % --------------------------------------------------------
    [rawFluxReward, phi_new] = cilia_ball_reward_continuous(loggedSignals.phi, action, P, opts.stepScale);

    rewardFlux = opts.fluxStepScale * rawFluxReward;

    % --------------------------------------------------------
    % 3. virtual x update
    % --------------------------------------------------------
    xOld = loggedSignals.xVirt;
    xNew = xOld + rawFluxReward;

    reward = -opts.stepPenalty + rewardFlux;

    % --------------------------------------------------------
    % 4. update span / arm state relative to current anchor
    % --------------------------------------------------------
    Xnew = position_from_angle(phi_new(:), P);
    xTipNew = Xnew(opts.xDispPointIndex, 1);

    xTipMinNew = min(loggedSignals.xTipMinSinceAnchor, xTipNew);
    xTipMaxNew = max(loggedSignals.xTipMaxSinceAnchor, xTipNew);
    tipSpanNew = xTipMaxNew - xTipMinNew;

    firstArm = (~loggedSignals.armed) && (tipSpanNew >= opts.spanArmThreshold);
    armedNew = loggedSignals.armed || firstArm;

    if firstArm
        reward = reward + opts.armBonus;
    end

    % --------------------------------------------------------
    % 5. closure / transport shaping (gated once armed)
    % --------------------------------------------------------
    cycleFluxOld = loggedSignals.xVirt - loggedSignals.xVirt_anchor;
    cycleFluxNew = xNew - loggedSignals.xVirt_anchor;

    closureDistOld = norm(loggedSignals.phi(:) - loggedSignals.phi_anchor(:));
    closureFracOld = closureDistOld / max(norm(2*P.phimax(:)), 1e-12);

    closureDistNew = norm(phi_new(:) - loggedSignals.phi_anchor(:));
    closureFracNew = closureDistNew / max(norm(2*P.phimax(:)), 1e-12);

    closeScoreOld = exp(-(closureFracOld / opts.sigmaClosureFrac)^2);
    closeScoreNew = exp(-(closureFracNew / opts.sigmaClosureFrac)^2);

    if armedNew
        rewardClosure = opts.lambdaClosure * max(0, closeScoreNew - closeScoreOld);
        rewardTransport = opts.lambdaTransport * max(0, cycleFluxNew - cycleFluxOld);
        reward = reward + rewardClosure + rewardTransport;
    end

    % --------------------------------------------------------
    % 6. cycle completion test relative to current anchor
    % --------------------------------------------------------
    stepsSinceAnchorNew = loggedSignals.stepsSinceAnchor + 1;

    cycleClosed = armedNew ...
        && (stepsSinceAnchorNew >= opts.minCycleSteps) ...
        && (closureFracNew <= opts.closeThreshold) ...
        && (cycleFluxNew >= opts.fluxThreshold);

    if cycleClosed
        reward = reward + opts.lambdaCycle * cycleFluxNew;
    end

    % --------------------------------------------------------
    % 7. advance main state
    % --------------------------------------------------------
    phi_old = loggedSignals.phi(:);

    loggedSignals.t = loggedSignals.t + 1;
    loggedSignals.phi_prev = phi_old;
    loggedSignals.phi = phi_new(:);
    loggedSignals.xVirt = xNew;

    % --------------------------------------------------------
    % 8. either reset anchor on successful cycle or continue it
    % --------------------------------------------------------
    if cycleClosed && opts.resetAnchorOnCycle
        loggedSignals.phi_anchor = phi_new(:);
        loggedSignals.xVirt_anchor = xNew;
        loggedSignals.stepsSinceAnchor = 0;
        loggedSignals.armed = false;
        loggedSignals.xTipMinSinceAnchor = xTipNew;
        loggedSignals.xTipMaxSinceAnchor = xTipNew;
    else
        loggedSignals.stepsSinceAnchor = stepsSinceAnchorNew;
        loggedSignals.armed = armedNew;
        loggedSignals.xTipMinSinceAnchor = xTipMinNew;
        loggedSignals.xTipMaxSinceAnchor = xTipMaxNew;
    end

    % --------------------------------------------------------
    % 9. episode termination
    % --------------------------------------------------------
    if opts.terminateOnCycle
        isDone = (loggedSignals.t >= opts.maxSteps) || cycleClosed;
    else
        isDone = (loggedSignals.t >= opts.maxSteps);
    end

    % --------------------------------------------------------
    % 10. next observation
    % --------------------------------------------------------
    nextObservation = local_getObservation(loggedSignals, P, opts);
end

function obs = local_getObservation(loggedSignals, P, opts)
    phi  = loggedSignals.phi(:);
    dphi = loggedSignals.phi(:) - loggedSignals.phi_prev(:);

    closureDist = norm(phi - loggedSignals.phi_anchor(:));
    closureFrac = closureDist / max(norm(2*P.phimax(:)), 1e-12);

    armedFlag = double(loggedSignals.armed);

    tipSpanSinceAnchor = loggedSignals.xTipMaxSinceAnchor - loggedSignals.xTipMinSinceAnchor;
    spanFrac = min(1, tipSpanSinceAnchor / max(opts.spanArmThreshold, 1e-12));

    cycleFlux = loggedSignals.xVirt - loggedSignals.xVirt_anchor;
    cycleProgress = min(1, loggedSignals.stepsSinceAnchor / max(opts.minCycleSteps, 1));

    obs = [phi; dphi; closureFrac; armedFlag; spanFrac; cycleFlux; cycleProgress];
end