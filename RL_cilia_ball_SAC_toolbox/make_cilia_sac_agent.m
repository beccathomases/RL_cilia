function agent = make_cilia_sac_agent(env)
% MAKE_CILIA_SAC_AGENT
% Build a small SAC agent for the 2-ball cilia RL Toolbox environment.

    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);

    obsDim = prod(obsInfo.Dimension);
    actDim = prod(actInfo.Dimension);

    % ------------------------------------------------------------
    % Actor network
    %
    % IMPORTANT for continuous SAC in RL Toolbox:
    % - use rlContinuousGaussianActor
    % - actor should output mean and standard deviation information
    % - do NOT put tanh/scaling at the end of the mean path; RL Toolbox
    %   handles action scaling automatically
    % - std path should end in something nonnegative, e.g. relu
    % ------------------------------------------------------------

    commonPath = [
        featureInputLayer(obsDim, Name="obs")
        fullyConnectedLayer(32, Name="fc1")
        reluLayer(Name="relu1")
        fullyConnectedLayer(32, Name="fc2")
        reluLayer(Name="relu2")
    ];

    meanPath = [
        fullyConnectedLayer(actDim, Name="meanFC")
    ];

    stdPath = [
        fullyConnectedLayer(actDim, Name="stdFC")
        reluLayer(Name="stdRelu")
    ];

    actorLG = layerGraph(commonPath);
    actorLG = addLayers(actorLG, meanPath);
    actorLG = addLayers(actorLG, stdPath);

    actorLG = connectLayers(actorLG, "relu2", "meanFC");
    actorLG = connectLayers(actorLG, "relu2", "stdFC");

    actorNet = dlnetwork(actorLG);

    actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
        ObservationInputNames="obs", ...
        ActionMeanOutputNames="meanFC", ...
        ActionStandardDeviationOutputNames="stdRelu");

    % ------------------------------------------------------------
    % Critic network template
    %
    % For continuous-action SAC, critics are Q(S,A) objects created with
    % rlQValueFunction.
    % ------------------------------------------------------------

    statePath = [
        featureInputLayer(obsDim, Name="state")
        fullyConnectedLayer(32, Name="stateFC")
    ];

    actionPath = [
        featureInputLayer(actDim, Name="action")
        fullyConnectedLayer(32, Name="actionFC")
    ];

    commonCriticPath = [
        additionLayer(2, Name="add")
        reluLayer(Name="relu1")
        fullyConnectedLayer(32, Name="fc2")
        reluLayer(Name="relu2")
        fullyConnectedLayer(1, Name="qOut")
    ];

    criticLG1 = layerGraph();
    criticLG1 = addLayers(criticLG1, statePath);
    criticLG1 = addLayers(criticLG1, actionPath);
    criticLG1 = addLayers(criticLG1, commonCriticPath);

    criticLG1 = connectLayers(criticLG1, "stateFC", "add/in1");
    criticLG1 = connectLayers(criticLG1, "actionFC", "add/in2");

    criticNet1 = dlnetwork(criticLG1);

    % Build critic 2 separately so it has different initial weights
    criticLG2 = layerGraph();
    criticLG2 = addLayers(criticLG2, statePath);
    criticLG2 = addLayers(criticLG2, actionPath);
    criticLG2 = addLayers(criticLG2, commonCriticPath);

    criticLG2 = connectLayers(criticLG2, "stateFC", "add/in1");
    criticLG2 = connectLayers(criticLG2, "actionFC", "add/in2");

    criticNet2 = dlnetwork(criticLG2);

    critic1 = rlQValueFunction(criticNet1, obsInfo, actInfo, ...
        ObservationInputNames="state", ...
        ActionInputNames="action");

    critic2 = rlQValueFunction(criticNet2, obsInfo, actInfo, ...
        ObservationInputNames="state", ...
        ActionInputNames="action");

    % ------------------------------------------------------------
    % SAC agent options
    % ------------------------------------------------------------

    agentOpts = rlSACAgentOptions( ...
        SampleTime=1, ...
        DiscountFactor=0.99, ...
        ExperienceBufferLength=1e5, ...
        MiniBatchSize=128);

    % Optional: adjust learn rates
    agentOpts.ActorOptimizerOptions.LearnRate = 1e-3;
    agentOpts.CriticOptimizerOptions(1).LearnRate = 1e-3;
    agentOpts.CriticOptimizerOptions(2).LearnRate = 1e-3;

    % ------------------------------------------------------------
    % Create SAC agent
    % ------------------------------------------------------------

    agent = rlSACAgent(actor, [critic1 critic2], agentOpts);
end