function [targetCritic1, targetCritic2] = soft_update_targets(targetCritic1, targetCritic2, critic1, critic2, tau)

    targetCritic1 = soft_update_one_target(targetCritic1, critic1, tau);
    targetCritic2 = soft_update_one_target(targetCritic2, critic2, tau);
end