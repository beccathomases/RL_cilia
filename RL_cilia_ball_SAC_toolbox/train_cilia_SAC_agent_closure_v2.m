% train_cilia_SAC_agent_closure_transport.m
%
% Closure-first hybrid SAC:
%   - same reward as current closure+transport version
%   - same reset/step/env behavior
%   - BUT improved reset pool for mixed_demo:
%       * tabular cycle states
%       * seed-3 SAC tail states
%
% Change only "seedin" to run multiple seeds.

clear; clc; close all;
seedin = 3;

% ------------------------------------------------------------
% Paths
% ------------------------------------------------------------
addpath('../RL_cilia_ball_tabular/');
addpath('../RL_cilia_ball_tabular_RUNS/');

demoDir = fullfile('..','RL_cilia_ball_tabular_RUNS');
if ~isfolder(demoDir)
    error('Demo directory not found: %s', demoDir);
end

fprintf('Using demo directory:\n%s\n', demoDir);

% ============================================================
% USER SETTINGS
% ============================================================

demoSeeds = 1:10;

% tabular-run parameters
nEpisodes_tab = 50000;
alpha0_tab    = 0.99;
epsilon0_tab  = 0.75;
gamma_tab     = 0.99;

% SAC training settings
onlineEpisodes = 400;
maxStepsPerEpisode = 250;
resetMode = 'mixed_demo';
stepScaleFactor = 0.5;

% mixed demo reset settings
pDemoReset = 0.8;
demoNoiseScaleFactor = 0.15;

% demo replay settings
demoLoopsPerSeed = 20;

% offline warm-start settings
offlineEpochs = 5;
offlineStepsPerEpoch = 400;

% ------------------------------------------------------------
% Reward settings  (UNCHANGED)
% ------------------------------------------------------------
stepPenalty      = 0.0010;
fluxStepScale    = 0.0020;

lambdaClosure    = 0.10;
lambdaTransport  = 0.05;
sigmaClosureFrac = 0.15;

spanArmThreshold   = 0.45;
closeThreshold     = 0.18;
minCycleSteps      = 10;
fluxThreshold      = 0.02;
lambdaCycle        = 4.0;
armBonus           = 0.0;
resetAnchorOnCycle = true;

terminateOnCycle = false;

onlineEpisodesTotal = 400;
episodesPerChunk    = 50;
nChunks = onlineEpisodesTotal / episodesPerChunk;

checkpointDir = fullfile(saveDir, sprintf('checkpoints_seed%03d', seed));
if ~exist(checkpointDir, "dir")
    mkdir(checkpointDir);
end

% ------------------------------------------------------------
% NEW: extra reset-pool source from promising SAC seed
% ------------------------------------------------------------
useSeed3ResetPool = true;
seed3EvalFile = "seed3_closure_transport_eval.mat";

% how to sample seed-3 rollout states
seed3BurnIn = 300;
seed3Stride = 5;

% relative weighting inside demoPhiList
tabularWeight = 2;
seed3Weight   = 3;

% save / RNG
saveDir = "saved_agents";
if ~exist(saveDir,"dir")
    mkdir(saveDir);
end

seed = seedin;
rng(seed,"twister");

% ============================================================
% BUILD TABULAR HELPER AND DEMO DATA FIRST
% ============================================================

P = setdefaultparams_ciliaball;
envDisc = ciliaBallTabularEnv(P, struct('reset_mode','fixed','precompute',false));

stepScale = stepScaleFactor * P.dphi(:);
demoNoiseScale = demoNoiseScaleFactor * P.dphi(:);

repeatVec = P.dphi(:) ./ stepScale;

if any(abs(repeatVec - round(repeatVec)) > 1e-10)
    error(['Current stepScale does not divide P.dphi by an integer factor. ' ...
           'Cannot build simple repeated-action demonstrations.']);
end

if abs(repeatVec(1) - repeatVec(2)) > 1e-10
    error(['Current stepScale gives different repeat counts for the two hinges. ' ...
           'This script assumes the same repeat count in both coordinates.']);
end

repeatCount = round(repeatVec(1));
fprintf('Using repeatCount = %d continuous substeps per tabular step.\n', repeatCount);

% ============================================================
% COMMON OPTIONS FOR DEMO STEP LOGIC
% ============================================================

commonOpts = struct();
commonOpts.maxSteps = maxStepsPerEpisode;
commonOpts.stepScaleFactor = stepScaleFactor;
commonOpts.stepScale = stepScale;
commonOpts.stepPenalty = stepPenalty;
commonOpts.fluxStepScale = fluxStepScale;
commonOpts.lambdaClosure = lambdaClosure;
commonOpts.lambdaTransport = lambdaTransport;
commonOpts.sigmaClosureFrac = sigmaClosureFrac;
commonOpts.spanArmThreshold = spanArmThreshold;
commonOpts.closeThreshold = closeThreshold;
commonOpts.minCycleSteps = minCycleSteps;
commonOpts.fluxThreshold = fluxThreshold;
commonOpts.lambdaCycle = lambdaCycle;
commonOpts.armBonus = armBonus;
commonOpts.resetAnchorOnCycle = resetAnchorOnCycle;
commonOpts.terminateOnCycle = terminateOnCycle;
commonOpts.xDispPointIndex = P.N;

% ============================================================
% BUILD DEMONSTRATION BUFFER AND TABULAR RESET STATES
% ============================================================

expBatch = struct('Observation',{},'Action',{},'Reward',{},'NextObservation',{},'IsDone',{});
demoPhiList = zeros(2,0);

for sIdx = 1:length(demoSeeds)
    demoSeed = demoSeeds(sIdx);

    cycleName = sprintf('cycle4_run_%d_g%1.2f_eps0%1.2f_alp0%1.2f_nEpisode%d.mat', ...
        demoSeed, gamma_tab, epsilon0_tab, alpha0_tab, nEpisodes_tab);

    cycleFile = fullfile(demoDir, cycleName);

    fprintf('Looking for demo file:\n%s\n', cycleFile);

    if ~isfile(cycleFile)
        warning('Could not find %s. Skipping this demo seed.', cycleFile);
        continue
    end

    S = load(cycleFile);
    if ~isfield(S,'cycle_states')
        warning('File %s does not contain cycle_states. Skipping.', cycleFile);
        continue
    end

    cycle_states = S.cycle_states(:).';

    if numel(cycle_states) >= 2 && cycle_states(1) == cycle_states(end)
        cycle_core = cycle_states(1:end-1);
    else
        cycle_core = cycle_states;
    end

    K = numel(cycle_core);
    if K < 2
        warning('Cycle in %s is too short. Skipping.', cycleFile);
        continue
    end

    fprintf('Using demo seed %d with cycle length %d\n', demoSeed, K);

    % add tabular cycle states to reset pool
    for j = 1:K
        phiDemo = envDisc.sub2phi(envDisc.state2sub(cycle_core(j)));
        demoPhiList(:,end+1) = phiDemo(:); %#ok<SAGROW>
    end

    % build replay demonstrations from tabular cycle
    for loopIdx = 1:demoLoopsPerSeed

        s0 = cycle_core(1);
        phi0 = envDisc.sub2phi(envDisc.state2sub(s0));
        phi0 = phi0(:);

        demoOpts = commonOpts;
        demoOpts.maxSteps = K * repeatCount;

        logged = local_init_loggedSignals(phi0, P, demoOpts);
        obs = local_getObservation(logged, P, demoOpts);

        doneEpisode = false;

        for j = 1:K
            if doneEpisode
                break
            end

            s_curr = cycle_core(j);
            if j == K
                s_next = cycle_core(1);
            else
                s_next = cycle_core(j+1);
            end

            a_disc = infer_action_vector(envDisc, s_curr, s_next);
            a_cont = a_disc(:);

            for m = 1:repeatCount
                if doneEpisode
                    break
                end

                thisObs = obs;
                [nextObs, reward, isDone, logged] = cilia_stepFcn(a_cont, logged, P, demoOpts);

                expOne.Observation = {thisObs};
                expOne.Action = {a_cont};
                expOne.Reward = reward;
                expOne.NextObservation = {nextObs};
                expOne.IsDone = isDone;

                expBatch(end+1,1) = expOne; %#ok<SAGROW>
                obs = nextObs;

                if isDone
                    doneEpisode = true;
                end
            end
        end
    end
end

if isempty(expBatch)
    error('No demonstration experiences were generated.');
end

if isempty(demoPhiList)
    error('No tabular demoPhiList states were built.');
end

fprintf('Built %d demonstration experiences.\n', numel(expBatch));
fprintf('Built tabular demoPhiList with %d phase points.\n', size(demoPhiList,2));

demoPhiList_tabular = demoPhiList;

% ============================================================
% APPEND SEED-3 RESET POOL STATES
% ============================================================

if useSeed3ResetPool
    R = load(seed3EvalFile);

    if isfield(R,'results') && isfield(R.results,'phiHist')
        phiSeed3 = R.results.phiHist;
    elseif isfield(R,'phiHist')
        phiSeed3 = R.phiHist;
    else
        error('Could not find phiHist in seed3EvalFile.');
    end

    if size(phiSeed3,1) ~= 2
        error('Expected phiHist to be 2 x N in seed3EvalFile.');
    end

    idx0 = min(seed3BurnIn + 1, size(phiSeed3,2));
    seed3PhiList = phiSeed3(:, idx0:seed3Stride:end);

    % clip to legal box
    seed3PhiList = max(-P.phimax(:), min(P.phimax(:), seed3PhiList));

    % weighted combined reset pool
    demoPhiList = [ ...
        repmat(demoPhiList_tabular, 1, tabularWeight), ...
        repmat(seed3PhiList,       1, seed3Weight) ...
        ];

    fprintf('Using combined demoPhiList:\n');
    fprintf('  tabular points: %d\n', size(demoPhiList_tabular,2));
    fprintf('  seed3 points:   %d\n', size(seed3PhiList,2));
    fprintf('  combined pool:  %d\n', size(demoPhiList,2));
else
    demoPhiList = demoPhiList_tabular;
end

% ============================================================
% BUILD ENVIRONMENT AND AGENT
% ============================================================

envOpts = struct();
envOpts.maxSteps = maxStepsPerEpisode;
envOpts.resetMode = resetMode;
envOpts.stepScaleFactor = stepScaleFactor;
envOpts.pDemoReset = pDemoReset;
envOpts.demoPhiList = demoPhiList;
envOpts.demoNoiseScale = demoNoiseScale;

envOpts.stepPenalty = stepPenalty;
envOpts.fluxStepScale = fluxStepScale;
envOpts.lambdaClosure = lambdaClosure;
envOpts.lambdaTransport = lambdaTransport;
envOpts.sigmaClosureFrac = sigmaClosureFrac;
envOpts.spanArmThreshold = spanArmThreshold;
envOpts.closeThreshold = closeThreshold;
envOpts.minCycleSteps = minCycleSteps;
envOpts.fluxThreshold = fluxThreshold;
envOpts.lambdaCycle = lambdaCycle;
envOpts.armBonus = armBonus;
envOpts.resetAnchorOnCycle = resetAnchorOnCycle;
envOpts.terminateOnCycle = terminateOnCycle;
envOpts.xDispPointIndex = P.N;

[env, obsInfo, actInfo, P, opts] = make_cilia2ball_rltoolbox_env(envOpts); %#ok<ASGLU>
agent = make_cilia_sac_agent(env);

agent.AgentOptions.ResetExperienceBufferBeforeTraining = false;
agent.AgentOptions.DiscountFactor = 0.995;
agent.AgentOptions.NumStepsToLookAhead = 3;

% ============================================================
% APPEND DEMOS TO REPLAY BUFFER
% ============================================================

validateExperience(agent.ExperienceBuffer, expBatch);
append(agent.ExperienceBuffer, expBatch);

fprintf('Appended demonstrations to replay buffer.\n');
demoBufferPreview = allExperiences(agent.ExperienceBuffer);
fprintf('Replay buffer now contains %d experiences.\n', numel(demoBufferPreview));

% ============================================================
% OFFLINE WARM-START FROM DEMO BUFFER
% ============================================================

offlineOpts = rlTrainingFromDataOptions;
offlineOpts.MaxEpochs = offlineEpochs;
offlineOpts.NumStepsPerEpoch = offlineStepsPerEpoch;
offlineOpts.Plots = "none";

fprintf('\nStarting offline warm-start from demonstrations...\n');
offlineStats = trainFromData(agent, offlineOpts);
fprintf('Offline warm-start finished.\n');

% ============================================================
% ONLINE SAC TRAINING WITH CHECKPOINTS
% ============================================================

allChunkStats = cell(nChunks,1);

for ichunk = 1:nChunks
    fprintf('\n==============================\n');
    fprintf('Starting chunk %d / %d\n', ichunk, nChunks);
    fprintf('Episodes this chunk: %d\n', episodesPerChunk);
    fprintf('Total episodes after chunk: %d\n', ichunk * episodesPerChunk);
    fprintf('==============================\n');

    trainOpts = rlTrainingOptions( ...
        MaxEpisodes=episodesPerChunk, ...
        MaxStepsPerEpisode=opts.maxSteps, ...
        Verbose=true, ...
        Plots="none", ...
        StopTrainingCriteria="EpisodeCount", ...
        StopTrainingValue=episodesPerChunk);

    statsChunk = train(agent, env, trainOpts);
    allChunkStats{ichunk} = statsChunk;

    onlineEpisodes = onlineEpisodesTotal;
    checkpointName = sprintf( ...
        "ckpt_%s_chunk%02d_ep%04d.mat", ...
        baseName, ichunk, ichunk*episodesPerChunk);

    checkpointFile = fullfile(checkpointDir, checkpointName);

    save(saveFile, ...
    "agent", "trainingStats", "offlineStats", "allChunkStats", ...
     "statsChunk", "P", "opts", "seed", ...
        "ichunk", "episodesPerChunk", "onlineEpisodesTotal", ...
        "stepPenalty", "fluxStepScale", ...
        "lambdaClosure", "lambdaTransport", "sigmaClosureFrac", ...
        "spanArmThreshold", "closeThreshold", "minCycleSteps", ...
        "fluxThreshold", "lambdaCycle", "armBonus", ...
        "resetAnchorOnCycle", "terminateOnCycle", ...
        "demoDir", "demoPhiList");
   

    fprintf('Saved checkpoint to:\n%s\n', checkpointFile);
end

% ============================================================
% SAVE FINAL AGENT
% ============================================================

baseName = sprintf("csac_closureTransport_mixReset_sp%03d_fs%03d_lcl%03d_ltr%03d_sig%03d_arm%02d_cl%02d_fx%02d_lcy%03d_pd%02d_dn%02d_s%03d_e%d_t%d_ss%02d", ...
    round(1000*stepPenalty), ...
    round(1000*fluxStepScale), ...
    round(1000*lambdaClosure), ...
    round(1000*lambdaTransport), ...
    round(1000*sigmaClosureFrac), ...
    round(100*spanArmThreshold), ...
    round(100*closeThreshold), ...
    round(100*fluxThreshold), ...
    round(100*lambdaCycle), ...
    round(100*pDemoReset), ...
    round(100*demoNoiseScaleFactor), ...
    seed, onlineEpisodes, opts.maxSteps, ...
    round(100*stepScaleFactor));

saveFile = fullfile(saveDir, baseName + ".mat");

save(saveFile, ...
    "agent", "trainingStats", "offlineStats", ...
    "P", "opts", "seed", ...
    "demoSeeds", "demoLoopsPerSeed", ...
    "nEpisodes_tab", "alpha0_tab", "epsilon0_tab", "gamma_tab", ...
    "onlineEpisodes", "maxStepsPerEpisode", ...
    "resetMode", "stepScaleFactor", ...
    "pDemoReset", "demoNoiseScaleFactor", "demoNoiseScale", ...
    "offlineEpochs", "offlineStepsPerEpoch", ...
    "stepPenalty", "fluxStepScale", ...
    "lambdaClosure", "lambdaTransport", "sigmaClosureFrac", ...
    "spanArmThreshold", "closeThreshold", "minCycleSteps", ...
    "fluxThreshold", "lambdaCycle", "armBonus", ...
    "resetAnchorOnCycle", "terminateOnCycle", ...
    "useSeed3ResetPool", "seed3EvalFile", "seed3BurnIn", "seed3Stride", ...
    "tabularWeight", "seed3Weight", ...
    "demoDir", "demoPhiList", "demoPhiList_tabular");

fprintf('\nSaved final agent to %s\n', saveFile);

% ============================================================
% LOCAL HELPERS
% ============================================================

function aVec = infer_action_vector(envDisc, s_curr, s_next)
    sub_curr = envDisc.state2sub(s_curr);
    sub_next = envDisc.state2sub(s_next);
    aVec = sub_next - sub_curr;
    aVec = aVec(:);
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