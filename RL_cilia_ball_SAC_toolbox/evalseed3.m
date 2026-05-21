%% eval_seed3_tabular_anchor_returns.m
clear; clc; close all;
addpath('/Users/bthomases/Documents/Student_Projects/RL_cilia/RL_cilia_ball_SAC_prototype');
addpath('../RL_cilia_ball_tabular/');
addpath('../RL_cilia_ball_tabular_RUNS/');

%% ============================================================
% USER SETTINGS
% ============================================================

%agentFile = "saved_agents/csac_closureTransport_sp001_fs002_lcl100_ltr050_sig150_arm45_cl18_fx02_lcy400_s003_e400_t250_ss50.mat";
seedin=3;
agentFile = sprintf("saved_agents/csac_closureTransport_mixReset_sp001_fs002_lcl100_ltr050_sig150_arm45_cl18_fx02_lcy400_pd80_dn15_s%03d_e400_t250_ss50.mat",seedin);

% pick one trusted tabular cycle file
cycleFile = fullfile('..','RL_cilia_ball_tabular_RUNS', ...
    'cycle4_run_3_g0.99_eps00.75_alp00.99_nEpisode50000.mat');

startMode = "random";   % "random" or "manual"
randomStartSeed = 1;
phi0_manual = [0; -0.3];

totalSteps = 1500;
burnInSteps = 300;

% anchor-return settings
nAnchorSamples = 6;     % number of anchor points sampled from tabular cycle
hitRadius      = 0.08;  % count as "near anchor"
resetRadius    = 0.12;  % must leave this radius before a new hit counts
minHitLag      = 15;    % minimum step gap between hits to same anchor

saveStem = "seed3_tabular_anchor_returns";

%% ============================================================
% LOAD AGENT / PARAMS
% ============================================================

S = load(agentFile);
agent = S.agent;

if isfield(S,'P')
    P = S.P;
else
    P = setdefaultparams_ciliaball;
end

opts = S.opts;
opts = local_fill_default_opts(opts, P);
opts.maxSteps = totalSteps;

if isprop(agent,'UseExplorationPolicy')
    agent.UseExplorationPolicy = false;
end

%% ============================================================
% LOAD TABULAR CYCLE AND BUILD ANCHORS
% ============================================================

C = load(cycleFile);
if ~isfield(C,'cycle_states')
    error('cycleFile does not contain cycle_states.');
end

envDisc = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

cycle_states = C.cycle_states(:).';
if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
    cycle_core = cycle_states(1:end-1);
else
    cycle_core = cycle_states;
end

K = numel(cycle_core);
if K < 2
    error('Tabular cycle is too short.');
end

phiCycle = zeros(2,K);
for j = 1:K
    phiCycle(:,j) = envDisc.sub2phi(envDisc.state2sub(cycle_core(j)));
end

anchorIdx = unique(round(linspace(1, K, nAnchorSamples+1)));
anchorIdx(end) = [];  % remove duplicate endpoint
nAnchors = numel(anchorIdx);

anchorPhi = phiCycle(:,anchorIdx);

fprintf('Loaded tabular cycle of length %d.\n', K);
fprintf('Using %d anchor points.\n', nAnchors);

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

phiHist   = nan(2, totalSteps+1);
xVirtHist = nan(1, totalSteps+1);

phiHist(:,1) = logged.phi;
xVirtHist(1) = logged.xVirt;

% per-anchor hit state
wasOutside  = true(1,nAnchors);
lastHitStep = -inf(1,nAnchors);
lastHitX    = nan(1,nAnchors);

hitRecords = struct( ...
    'anchorID', {}, ...
    'cycleIndex', {}, ...
    'phiRef1', {}, ...
    'phiRef2', {}, ...
    'hitNumber', {}, ...
    'hitStep', {}, ...
    'hitDist', {}, ...
    'xVirtAtHit', {}, ...
    'deltaXSincePrev', {}, ...
    'lagSincePrev', {} );

%% ============================================================
% LONG ROLLOUT + ANCHOR RETURN DETECTION
% ============================================================

nSteps = 0;

for k = 1:totalSteps
    nSteps = k;

    action = local_getAgentAction(agent, obs);
    [nextObs, ~, isDone, loggedNext] = cilia_stepFcn(action, logged, P, opts);

    phiNow = loggedNext.phi(:);
    xNow   = loggedNext.xVirt;

    phiHist(:,k+1) = phiNow;
    xVirtHist(k+1) = xNow;

    if k >= burnInSteps
        for ia = 1:nAnchors
            phiRef = anchorPhi(:,ia);
            d = norm(phiNow - phiRef);

            if d >= resetRadius
                wasOutside(ia) = true;
            end

            isNewHit = (d <= hitRadius) && wasOutside(ia) && ((k - lastHitStep(ia)) >= minHitLag);

            if isNewHit
                if isnan(lastHitX(ia))
                    dx = NaN;
                    lag = NaN;
                    hitNumber = 1;
                else
                    dx = xNow - lastHitX(ia);
                    lag = k - lastHitStep(ia);
                    prevHits = sum([hitRecords.anchorID] == ia);
                    hitNumber = prevHits + 1;
                end

                rec.anchorID = ia;
                rec.cycleIndex = anchorIdx(ia);
                rec.phiRef1 = phiRef(1);
                rec.phiRef2 = phiRef(2);
                rec.hitNumber = hitNumber;
                rec.hitStep = k;
                rec.hitDist = d;
                rec.xVirtAtHit = xNow;
                rec.deltaXSincePrev = dx;
                rec.lagSincePrev = lag;

                hitRecords(end+1) = rec; %#ok<SAGROW>

                lastHitStep(ia) = k;
                lastHitX(ia) = xNow;
                wasOutside(ia) = false;
            end
        end
    end

    logged = loggedNext;
    obs = nextObs;

    if isDone
        break
    end
end

phiHist   = phiHist(:,1:nSteps+1);
xVirtHist = xVirtHist(1:nSteps+1);

fprintf('Completed rollout with %d steps.\n', nSteps);

%% ============================================================
% BUILD TABLES
% ============================================================

if isempty(hitRecords)
    warning('No anchor returns detected.');
    hitTable = table();
    summaryTable = table();
else
    hitTable = struct2table(hitRecords);

    summaryRows = struct( ...
        'anchorID', {}, ...
        'cycleIndex', {}, ...
        'phiRef1', {}, ...
        'phiRef2', {}, ...
        'nHits', {}, ...
        'nReturns', {}, ...
        'meanHitDist', {}, ...
        'medianLag', {}, ...
        'meanDeltaX', {}, ...
        'stdDeltaX', {}, ...
        'lastDeltaX', {}, ...
        'positiveReturnFrac', {} );

    for ia = 1:nAnchors
        Ti = hitTable(hitTable.anchorID == ia, :);

        if isempty(Ti)
            nHits = 0;
            nReturns = 0;
            meanHitDist = NaN;
            medianLag = NaN;
            meanDeltaX = NaN;
            stdDeltaX = NaN;
            lastDeltaX = NaN;
            positiveReturnFrac = NaN;
        else
            validDx = ~isnan(Ti.deltaXSincePrev);
            validLag = ~isnan(Ti.lagSincePrev);

            nHits = height(Ti);
            nReturns = sum(validDx);
            meanHitDist = mean(Ti.hitDist);

            if any(validLag)
                medianLag = median(Ti.lagSincePrev(validLag));
            else
                medianLag = NaN;
            end

            if any(validDx)
                dxv = Ti.deltaXSincePrev(validDx);
                meanDeltaX = mean(dxv);
                stdDeltaX = std(dxv);
                lastDeltaX = dxv(end);
                positiveReturnFrac = mean(dxv > 0);
            else
                meanDeltaX = NaN;
                stdDeltaX = NaN;
                lastDeltaX = NaN;
                positiveReturnFrac = NaN;
            end
        end

        s.anchorID = ia;
        s.cycleIndex = anchorIdx(ia);
        s.phiRef1 = anchorPhi(1,ia);
        s.phiRef2 = anchorPhi(2,ia);
        s.nHits = nHits;
        s.nReturns = nReturns;
        s.meanHitDist = meanHitDist;
        s.medianLag = medianLag;
        s.meanDeltaX = meanDeltaX;
        s.stdDeltaX = stdDeltaX;
        s.lastDeltaX = lastDeltaX;
        s.positiveReturnFrac = positiveReturnFrac;

        summaryRows(end+1) = s; %#ok<SAGROW>
    end

    summaryTable = struct2table(summaryRows);
    summaryTable = sortrows(summaryTable, {'nReturns','meanDeltaX','meanHitDist'}, {'descend','descend','ascend'});
end

disp(' ');
disp('==== Anchor summary table ====');
disp(summaryTable);

disp(' ');
disp('==== Detailed hit table ====');
disp(hitTable);

save(saveStem + ".mat", ...
    "summaryTable", "hitTable", "anchorIdx", "anchorPhi", ...
    "phiCycle", "phiHist", "xVirtHist", ...
    "agentFile", "cycleFile", "phi0", ...
    "hitRadius", "resetRadius", "minHitLag", ...
    "burnInSteps", "totalSteps");

writetable(summaryTable, saveStem + "_summary.csv");
writetable(hitTable, saveStem + "_hits.csv");

fprintf('\nSaved:\n');
fprintf('  %s\n', saveStem + ".mat");
fprintf('  %s\n', saveStem + "_summary.csv");
fprintf('  %s\n', saveStem + "_hits.csv");

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

function action = local_getAgentAction(agent, obs)
    a = getAction(agent, obs);
    if iscell(a)
        action = a{1};
    else
        action = a;
    end
    action = action(:);
end