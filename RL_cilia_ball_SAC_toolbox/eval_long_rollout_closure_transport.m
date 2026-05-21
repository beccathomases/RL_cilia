%% eval_long_rollout_closure_transport.m
clear; clc; close all;
addpath('/Users/bthomases/Documents/Student_Projects/RL_cilia/RL_cilia_ball_SAC_prototype');

%% ============================================================
% USER SETTINGS
% ============================================================
seedin= 3;                          
%agentFile = sprintf("saved_agents/csac_closureTransport_mixReset_sp001_fs002_lcl100_ltr050_sig150_arm45_cl18_fx02_lcy400_pd80_dn15_s%03d_e400_t250_ss50.mat",seedin);
agentFile = sprintf("saved_agents/csac_closureTransport_sp001_fs002_lcl100_ltr050_sig150_arm45_cl18_fx02_lcy400_s%03d_e400_t500_ss50.mat",seedin);%csac_closureTransport_mixReset_sp001_fs002_lcl100_ltr050_sig150_arm45_cl18_fx02_lcy400_pd80_dn15_s%03d_e400_t250_ss50.mat",seedin);

% change only this line for a different saved agent


startMode = "random";   % "random" or "manual"

randomStartSeed = 1;
phi0_manual = [0; -0.3];

totalSteps = 1500;

% retrace analysis settings
minLagPoint   = 20;
segmentLength = 60;
segmentGap    = 30;
compareStride = 2;
tailWindow    = 400;

% thresholds for "useful repeated path"
xGainTol    = 0.02;
fluxGainTol = 0.02;

plotEveryBody = 4;

%% ============================================================
% LOAD AGENT / PARAMS
% ============================================================

S = load(agentFile);

if ~isfield(S,'agent')
    error('agentFile does not contain variable ''agent''.');
end
agent = S.agent;

if isfield(S,'P')
    P = S.P;
else
    P = setdefaultparams_ciliaball;
end

if isfield(S,'opts')
    opts = S.opts;
else
    error('agentFile does not contain variable ''opts''.');
end

opts = local_fill_default_opts(opts, P);
opts.maxSteps = totalSteps;

if isprop(agent, 'UseExplorationPolicy')
    agent.UseExplorationPolicy = false;
end

%% ============================================================
% CHOOSE START
% ============================================================

switch lower(startMode)
    case 'random'
        rng(randomStartSeed, 'twister');
        phi0 = -P.phimax(:) + 2 * P.phimax(:) .* rand(2,1);

    case 'manual'
        phi0 = phi0_manual(:);

    otherwise
        error('Unknown startMode: %s', startMode);
end

phi0 = max(-P.phimax(:), min(P.phimax(:), phi0));

fprintf('\nStart mode: %s\n', startMode);
fprintf('phi0 = [%.6f, %.6f]\n', phi0(1), phi0(2));

%% ============================================================
% INITIALIZE ROLLOUT
% ============================================================

logged = local_init_loggedSignals(phi0, P, opts);
obs    = local_getObservation(logged, P, opts);

phiHist               = nan(2, totalSteps+1);
xVirtHist             = nan(1, totalSteps+1);
rewardHist            = nan(1, totalSteps);
rawFluxHist           = nan(1, totalSteps);
rewardStepPenaltyHist = nan(1, totalSteps);
rewardFluxHist        = nan(1, totalSteps);
rewardClosureHist     = nan(1, totalSteps);
rewardTransportHist   = nan(1, totalSteps);
rewardArmBonusHist    = nan(1, totalSteps);
rewardCycleHist       = nan(1, totalSteps);
armedHist             = nan(1, totalSteps+1);
closureFracHist       = nan(1, totalSteps+1);
bodyHist              = cell(1, totalSteps+1);

phiHist(:,1) = logged.phi;
xVirtHist(1) = logged.xVirt;
armedHist(1) = double(logged.armed);
closureFracHist(1) = local_closureFrac(logged.phi, logged.phi_anchor, P);

X0 = position_from_angle(logged.phi, P);
bodyHist{1} = [P.X0; X0];

%% ============================================================
% LONG ROLLOUT
% ============================================================

nSteps = 0;

for k = 1:totalSteps
    nSteps = k;

    action = local_getAgentAction(agent, obs);
    comp = local_reward_components(logged, action, P, opts);

    [nextObs, reward, isDone, loggedNext] = cilia_stepFcn(action, logged, P, opts);

    rewardHist(k)            = reward;
    rawFluxHist(k)           = comp.rawFlux;
    rewardStepPenaltyHist(k) = comp.rewardStepPenalty;
    rewardFluxHist(k)        = comp.rewardFlux;
    rewardClosureHist(k)     = comp.rewardClosure;
    rewardTransportHist(k)   = comp.rewardTransport;
    rewardArmBonusHist(k)    = comp.rewardArmBonus;
    rewardCycleHist(k)       = comp.rewardCycle;

    phiHist(:,k+1) = loggedNext.phi;
    xVirtHist(k+1) = loggedNext.xVirt;
    armedHist(k+1) = double(loggedNext.armed);
    closureFracHist(k+1) = local_closureFrac(loggedNext.phi, loggedNext.phi_anchor, P);

    Xnew = position_from_angle(loggedNext.phi, P);
    bodyHist{k+1} = [P.X0; Xnew];

    logged = loggedNext;
    obs = nextObs;

    if isDone
        break
    end
end

phiHist               = phiHist(:,1:nSteps+1);
xVirtHist             = xVirtHist(1:nSteps+1);
armedHist             = armedHist(1:nSteps+1);
closureFracHist       = closureFracHist(1:nSteps+1);
rewardHist            = rewardHist(1:nSteps);
rawFluxHist           = rawFluxHist(1:nSteps);
rewardStepPenaltyHist = rewardStepPenaltyHist(1:nSteps);
rewardFluxHist        = rewardFluxHist(1:nSteps);
rewardClosureHist     = rewardClosureHist(1:nSteps);
rewardTransportHist   = rewardTransportHist(1:nSteps);
rewardArmBonusHist    = rewardArmBonusHist(1:nSteps);
rewardCycleHist       = rewardCycleHist(1:nSteps);
bodyHist              = bodyHist(1:nSteps+1);

fprintf('Completed rollout with %d steps.\n', nSteps);

%% ============================================================
% RETRACE ANALYSIS
% ============================================================

nStates = size(phiHist,2);

% richer state for matching
stateHist = [phiHist; xVirtHist; closureFracHist; armedHist];

% pointwise revisit distance
pointRevisitDist = nan(1, nStates);
pointRevisitIdx  = nan(1, nStates);

for t = 1:nStates
    if t <= minLagPoint + 1
        continue
    end
    prevIdx = 1:(t-minLagPoint-1);
    diffs = vecnorm(stateHist(:,prevIdx) - stateHist(:,t), 2, 1);
    [dmin, imin] = min(diffs);
    pointRevisitDist(t) = dmin;
    pointRevisitIdx(t)  = prevIdx(imin);
end

% segment retrace metrics
segmentBestDist      = nan(1, nStates);
segmentBestPrevEnd   = nan(1, nStates);
segmentXVirtGain     = nan(1, nStates);
segmentRawFluxGain   = nan(1, nStates);
segmentCycleReward   = nan(1, nStates);
segmentUsefulness    = nan(1, nStates);

tailStart = max(1, nStates - tailWindow + 1);
currEndMin = max(2*segmentLength + segmentGap, tailStart + segmentLength - 1);

for tEnd = currEndMin : nStates
    currStart = tEnd - segmentLength + 1;
    currSeg = stateHist(:, currStart:tEnd);

    bestDist = inf;
    bestPrevEnd = nan;

    prevEnds = segmentLength : compareStride : (currStart - segmentGap);

    for pEnd = prevEnds
        pStart = pEnd - segmentLength + 1;
        if pStart < 1
            continue
        end

        prevSeg = stateHist(:, pStart:pEnd);
        d = mean(vecnorm(currSeg - prevSeg, 2, 1));

        if d < bestDist
            bestDist = d;
            bestPrevEnd = pEnd;
        end
    end

    segmentBestDist(tEnd) = bestDist;
    segmentBestPrevEnd(tEnd) = bestPrevEnd;

    xGain = xVirtHist(tEnd) - xVirtHist(currStart);
    rawFluxGain = sum(rawFluxHist(currStart:min(nSteps,tEnd-1)));
    cycReward = sum(rewardCycleHist(currStart:min(nSteps,tEnd-1)));

    segmentXVirtGain(tEnd)   = xGain;
    segmentRawFluxGain(tEnd) = rawFluxGain;
    segmentCycleReward(tEnd) = cycReward;

    usefulGate = double((xGain >= xGainTol) || (rawFluxGain >= fluxGainTol));

    if isfinite(bestDist) && usefulGate > 0
        segmentUsefulness(tEnd) = usefulGate * (xGain + rawFluxGain) / (1e-8 + bestDist);
    else
        segmentUsefulness(tEnd) = 0;
    end
end

tailCandidates = find((1:nStates) >= currEndMin & ~isnan(segmentBestDist));

if isempty(tailCandidates)
    error('No valid tail segment candidates. Try smaller segmentLength or smaller tailWindow.');
end

% best purely repeated segment in tail
[tailBestSegDist_repeat, idxLocal1] = min(segmentBestDist(tailCandidates));
bestCurrEnd_repeat = tailCandidates(idxLocal1);
bestPrevEnd_repeat = segmentBestPrevEnd(bestCurrEnd_repeat);

bestCurrStart_repeat = bestCurrEnd_repeat - segmentLength + 1;
bestPrevStart_repeat = bestPrevEnd_repeat - segmentLength + 1;

% best repeated AND transporting segment in tail
[bestUsefulScore, idxLocal2] = max(segmentUsefulness(tailCandidates));
bestCurrEnd_useful = tailCandidates(idxLocal2);
bestPrevEnd_useful = segmentBestPrevEnd(bestCurrEnd_useful);

bestCurrStart_useful = bestCurrEnd_useful - segmentLength + 1;
bestPrevStart_useful = bestPrevEnd_useful - segmentLength + 1;

tailXVirtGain_useful   = segmentXVirtGain(bestCurrEnd_useful);
tailRawFluxGain_useful = segmentRawFluxGain(bestCurrEnd_useful);
tailCycleReward_useful = segmentCycleReward(bestCurrEnd_useful);
tailBestSegDist_useful = segmentBestDist(bestCurrEnd_useful);

isUsefulRepeat = (bestUsefulScore > 0);

fprintf('\n---- Retrace summary ----\n');
fprintf('tailStart state index                = %d\n', tailStart);

fprintf('\nBest repeated tail segment:\n');
fprintf('  prev [%d,%d]  <->  curr [%d,%d]\n', ...
    bestPrevStart_repeat, bestPrevEnd_repeat, ...
    bestCurrStart_repeat, bestCurrEnd_repeat);
fprintf('  repeated-segment distance          = %.6e\n', tailBestSegDist_repeat);

fprintf('\nBest useful repeated tail segment:\n');
fprintf('  prev [%d,%d]  <->  curr [%d,%d]\n', ...
    bestPrevStart_useful, bestPrevEnd_useful, ...
    bestCurrStart_useful, bestCurrEnd_useful);
fprintf('  repeated-segment distance          = %.6e\n', tailBestSegDist_useful);
fprintf('  xVirt gain over current segment    = %.6f\n', tailXVirtGain_useful);
fprintf('  rawFlux gain over current segment  = %.6f\n', tailRawFluxGain_useful);
fprintf('  cycle reward over current segment  = %.6f\n', tailCycleReward_useful);
fprintf('  useful score                       = %.6e\n', bestUsefulScore);

fprintf('\nUseful transporting repeat found? %d\n', isUsefulRepeat);

%% ============================================================
% FIGURES
% ============================================================

figure('Name', 'PhasePlane_LongRetrace');
plot(phiHist(1,:), phiHist(2,:), '-', 'Color', [0.82 0.82 0.82], 'LineWidth', 1.0); hold on;
plot(phiHist(1,tailStart:end), phiHist(2,tailStart:end), 'b-', 'LineWidth', 1.8);
plot(phiHist(1,bestPrevStart_useful:bestPrevEnd_useful), ...
     phiHist(2,bestPrevStart_useful:bestPrevEnd_useful), ...
     'g-', 'LineWidth', 2.2);
plot(phiHist(1,bestCurrStart_useful:bestCurrEnd_useful), ...
     phiHist(2,bestCurrStart_useful:bestCurrEnd_useful), ...
     'm-', 'LineWidth', 2.2);
plot(phiHist(1,1), phiHist(2,1), 'ro', 'MarkerFaceColor','r', 'MarkerSize', 8);
plot(phiHist(1,end), phiHist(2,end), 'ko', 'MarkerFaceColor','k', 'MarkerSize', 8);
xlabel('\phi_1');
ylabel('\phi_2');
title(sprintf('Phase plane | best useful tail seg dist = %.4e | xGain = %.4f', ...
    tailBestSegDist_useful, tailXVirtGain_useful));
legend('full rollout', 'tail', 'best useful previous seg', 'best useful current seg', ...
    'start', 'end', 'Location','best');
grid on;

figure('Name', 'RetraceMetrics');
yyaxis left
plot(1:nStates, pointRevisitDist, '-', 'LineWidth', 1.0); hold on;
plot(1:nStates, segmentBestDist, '-', 'LineWidth', 1.5);
plot(1:nStates, segmentUsefulness, '-', 'LineWidth', 1.8);
ylabel('distance / usefulness');

yyaxis right
plot(1:nStates, xVirtHist, '-', 'LineWidth', 1.2);
ylabel('xVirt');

xline(tailStart, '--', 'tail start', 'LineWidth', 1.2);
xline(bestCurrEnd_useful, ':', 'best useful curr end', 'LineWidth', 1.2);

xlabel('state index');
title('Retrace metrics vs time');
legend('point revisit dist', 'segment retrace dist', 'segment usefulness', 'xVirt', ...
    'Location','best');
grid on;

figure('Name', 'BodyOverlay_BestRepeatedSegments');
hold on;

prevIdx = bestPrevStart_useful:plotEveryBody:bestPrevEnd_useful;
for j = 1:numel(prevIdx)
    XX = bodyHist{prevIdx(j)};
    frac = (j-1) / max(numel(prevIdx)-1,1);
    c = [0, 0.6 + 0.4*(1-frac), 0];
    plot(XX(:,1), XX(:,3), '-', 'Color', c, 'LineWidth', 1.2);
end

currIdx = bestCurrStart_useful:plotEveryBody:bestCurrEnd_useful;
for j = 1:numel(currIdx)
    XX = bodyHist{currIdx(j)};
    frac = (j-1) / max(numel(currIdx)-1,1);
    c = [0.6 + 0.4*(1-frac), 0, 0.6 + 0.4*(1-frac)];
    plot(XX(:,1), XX(:,3), '-', 'Color', c, 'LineWidth', 1.2);
end

XXp0 = bodyHist{bestPrevStart_useful};
XXp1 = bodyHist{bestPrevEnd_useful};
plot(XXp0(:,1), XXp0(:,3), 'g-', 'LineWidth', 2.5);
plot(XXp1(:,1), XXp1(:,3), 'g--', 'LineWidth', 2.0);

XXc0 = bodyHist{bestCurrStart_useful};
XXc1 = bodyHist{bestCurrEnd_useful};
plot(XXc0(:,1), XXc0(:,3), 'm-', 'LineWidth', 2.5);
plot(XXc1(:,1), XXc1(:,3), 'm--', 'LineWidth', 2.0);

axis equal;
xlabel('x');
ylabel('z');
title('Best useful repeated body-segment pair: previous (green), current (magenta)');
grid on;

%% ============================================================
% SAVE RESULTS
% ============================================================

results = struct();
results.agentFile = agentFile;
results.startMode = startMode;
results.phi0 = phi0;
results.totalSteps = totalSteps;
results.nSteps = nSteps;
results.minLagPoint = minLagPoint;
results.segmentLength = segmentLength;
results.segmentGap = segmentGap;
results.compareStride = compareStride;
results.tailWindow = tailWindow;
results.xGainTol = xGainTol;
results.fluxGainTol = fluxGainTol;

results.phiHist = phiHist;
results.xVirtHist = xVirtHist;
results.rewardHist = rewardHist;
results.rawFluxHist = rawFluxHist;
results.rewardStepPenaltyHist = rewardStepPenaltyHist;
results.rewardFluxHist = rewardFluxHist;
results.rewardClosureHist = rewardClosureHist;
results.rewardTransportHist = rewardTransportHist;
results.rewardArmBonusHist = rewardArmBonusHist;
results.rewardCycleHist = rewardCycleHist;
results.armedHist = armedHist;
results.closureFracHist = closureFracHist;

results.pointRevisitDist = pointRevisitDist;
results.pointRevisitIdx = pointRevisitIdx;
results.segmentBestDist = segmentBestDist;
results.segmentBestPrevEnd = segmentBestPrevEnd;
results.segmentXVirtGain = segmentXVirtGain;
results.segmentRawFluxGain = segmentRawFluxGain;
results.segmentCycleReward = segmentCycleReward;
results.segmentUsefulness = segmentUsefulness;

results.tailStart = tailStart;

results.bestPrevStart_repeat = bestPrevStart_repeat;
results.bestPrevEnd_repeat = bestPrevEnd_repeat;
results.bestCurrStart_repeat = bestCurrStart_repeat;
results.bestCurrEnd_repeat = bestCurrEnd_repeat;
results.tailBestSegDist_repeat = tailBestSegDist_repeat;

results.bestPrevStart_useful = bestPrevStart_useful;
results.bestPrevEnd_useful = bestPrevEnd_useful;
results.bestCurrStart_useful = bestCurrStart_useful;
results.bestCurrEnd_useful = bestCurrEnd_useful;
results.tailBestSegDist_useful = tailBestSegDist_useful;
results.tailXVirtGain_useful = tailXVirtGain_useful;
results.tailRawFluxGain_useful = tailRawFluxGain_useful;
results.tailCycleReward_useful = tailCycleReward_useful;
results.bestUsefulScore = bestUsefulScore;
results.isUsefulRepeat = isUsefulRepeat;

save('eval_long_rollout_closure_transport_results.mat', 'results');
fprintf('\nSaved results to eval_long_rollout_closure_transport_results.mat\n');

%% ============================================================
% LOCAL FUNCTIONS
% ============================================================

function opts = local_fill_default_opts(opts, P)

    if ~isfield(opts,'maxSteps'), opts.maxSteps = 100; end
    if ~isfield(opts,'stepScaleFactor'), opts.stepScaleFactor = 0.5; end
    if ~isfield(opts,'stepScale'), opts.stepScale = opts.stepScaleFactor * P.dphi(:); end

    if ~isfield(opts,'stepPenalty'), opts.stepPenalty = 0.0010; end
    if ~isfield(opts,'fluxStepScale'), opts.fluxStepScale = 0.0020; end

    if ~isfield(opts,'lambdaClosure'), opts.lambdaClosure = 0.10; end
    if ~isfield(opts,'lambdaTransport'), opts.lambdaTransport = 0.05; end
    if ~isfield(opts,'sigmaClosureFrac'), opts.sigmaClosureFrac = 0.15; end

    if ~isfield(opts,'spanArmThreshold'), opts.spanArmThreshold = 0.45; end
    if ~isfield(opts,'closeThreshold'), opts.closeThreshold = 0.18; end
    if ~isfield(opts,'minCycleSteps'), opts.minCycleSteps = 10; end
    if ~isfield(opts,'fluxThreshold'), opts.fluxThreshold = 0.02; end
    if ~isfield(opts,'lambdaCycle'), opts.lambdaCycle = 4.0; end
    if ~isfield(opts,'armBonus'), opts.armBonus = 0.0; end
    if ~isfield(opts,'resetAnchorOnCycle'), opts.resetAnchorOnCycle = true; end

    if ~isfield(opts,'terminateOnCycle'), opts.terminateOnCycle = false; end
    if ~isfield(opts,'xDispPointIndex'), opts.xDispPointIndex = P.N; end

    opts.stepScale = opts.stepScale(:);
end

function logged = local_init_loggedSignals(phi0, P, opts)
    logged = struct();
    logged.t = 0;
    logged.phi = phi0(:);
    logged.phi_prev = phi0(:);

    logged.xVirt = 0;
    logged.phi_anchor = phi0(:);
    logged.xVirt_anchor = 0;
    logged.stepsSinceAnchor = 0;
    logged.armed = false;

    X0 = position_from_angle(phi0(:), P);
    xTip0 = X0(opts.xDispPointIndex,1);
    logged.xTipMinSinceAnchor = xTip0;
    logged.xTipMaxSinceAnchor = xTip0;
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

function closureFrac = local_closureFrac(phi, phiAnchor, P)
    closureDist = norm(phi(:) - phiAnchor(:));
    closureFrac = closureDist / max(norm(2*P.phimax(:)), 1e-12);
end

function action = local_getAgentAction(agent, obs)
    a = getAction(agent, obs);
    if iscell(a)
        action = a{1};
    else
        action = a;
    end
    action = action(:);
end

function comp = local_reward_components(logged, action, P, opts)

    action = max(-ones(2,1), min(ones(2,1), action(:)));

    [rawFlux, phi_new] = cilia_ball_reward_continuous(logged.phi, action, P, opts.stepScale);

    rewardFlux = opts.fluxStepScale * rawFlux;
    rewardStepPenalty = -opts.stepPenalty;

    xOld = logged.xVirt;
    xNew = xOld + rawFlux;

    Xnew = position_from_angle(phi_new(:), P);
    xTipNew = Xnew(opts.xDispPointIndex,1);

    xTipMinNew = min(logged.xTipMinSinceAnchor, xTipNew);
    xTipMaxNew = max(logged.xTipMaxSinceAnchor, xTipNew);
    tipSpanNew = xTipMaxNew - xTipMinNew;

    firstArm = (~logged.armed) && (tipSpanNew >= opts.spanArmThreshold);
    armedNew = logged.armed || firstArm;

    rewardArmBonus = 0;
    if firstArm
        rewardArmBonus = opts.armBonus;
    end

    cycleFluxOld = logged.xVirt - logged.xVirt_anchor;
    cycleFluxNew = xNew - logged.xVirt_anchor;

    closureDistOld = norm(logged.phi(:) - logged.phi_anchor(:));
    closureFracOld = closureDistOld / max(norm(2*P.phimax(:)), 1e-12);

    closureDistNew = norm(phi_new(:) - logged.phi_anchor(:));
    closureFracNew = closureDistNew / max(norm(2*P.phimax(:)), 1e-12);

    closeScoreOld = exp(-(closureFracOld / opts.sigmaClosureFrac)^2);
    closeScoreNew = exp(-(closureFracNew / opts.sigmaClosureFrac)^2);

    rewardClosure = 0;
    rewardTransport = 0;
    if armedNew
        rewardClosure = opts.lambdaClosure * max(0, closeScoreNew - closeScoreOld);
        rewardTransport = opts.lambdaTransport * max(0, cycleFluxNew - cycleFluxOld);
    end

    stepsSinceAnchorNew = logged.stepsSinceAnchor + 1;

    cycleClosed = armedNew ...
        && (stepsSinceAnchorNew >= opts.minCycleSteps) ...
        && (closureFracNew <= opts.closeThreshold) ...
        && (cycleFluxNew >= opts.fluxThreshold);

    rewardCycle = 0;
    if cycleClosed
        rewardCycle = opts.lambdaCycle * cycleFluxNew;
    end

    rewardTotalApprox = rewardStepPenalty + rewardFlux + rewardClosure + ...
                        rewardTransport + rewardArmBonus + rewardCycle;

    comp.rawFlux = rawFlux;
    comp.rewardStepPenalty = rewardStepPenalty;
    comp.rewardFlux = rewardFlux;
    comp.rewardClosure = rewardClosure;
    comp.rewardTransport = rewardTransport;
    comp.rewardArmBonus = rewardArmBonus;
    comp.rewardCycle = rewardCycle;
    comp.rewardTotalApprox = rewardTotalApprox;

    comp.firstArm = firstArm;
    comp.armedNew = armedNew;
    comp.cycleFluxOld = cycleFluxOld;
    comp.cycleFluxNew = cycleFluxNew;
    comp.closureFracNew = closureFracNew;
    comp.stepsSinceAnchorNew = stepsSinceAnchorNew;
    comp.cycleClosed = cycleClosed;
end