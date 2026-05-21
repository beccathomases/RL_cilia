function [initialObservation, loggedSignals] = cilia_resetFcn(P, opts)
% CILIA_RESETFCN
%
% Closure + transport SAC version.

    switch lower(opts.resetMode)

        case 'fixed'
            phi0 = [0; -P.phimax(2)];

        case 'random'
            phi0 = -P.phimax(:) + 2 * P.phimax(:) .* rand(2,1);

        case 'mixed_demo'
            useDemo = ~isempty(opts.demoPhiList) && (rand < opts.pDemoReset);

            if useDemo
                idx = randi(size(opts.demoPhiList,2));
                phi0 = opts.demoPhiList(:,idx);
                phi0 = phi0 + opts.demoNoiseScale(:) .* randn(2,1);
                phi0 = max(-P.phimax(:), min(P.phimax(:), phi0));
            else
                phi0 = -P.phimax(:) + 2 * P.phimax(:) .* rand(2,1);
            end

        otherwise
            error('Unknown reset mode: %s', opts.resetMode);
    end

    loggedSignals = struct();
    loggedSignals.t = 0;

    loggedSignals.phi = phi0(:);
    loggedSignals.phi_prev = phi0(:);

    % virtual x-position
    loggedSignals.xVirt = 0;

    % current cycle anchor
    loggedSignals.phi_anchor = phi0(:);
    loggedSignals.xVirt_anchor = 0;
    loggedSignals.stepsSinceAnchor = 0;
    loggedSignals.armed = false;

    % span since anchor
    X0 = position_from_angle(phi0(:), P);
    xTip0 = X0(opts.xDispPointIndex, 1);
    loggedSignals.xTipMinSinceAnchor = xTip0;
    loggedSignals.xTipMaxSinceAnchor = xTip0;

    initialObservation = local_getObservation(loggedSignals, P, opts);
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