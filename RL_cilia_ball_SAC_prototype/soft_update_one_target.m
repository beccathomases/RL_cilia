function target = soft_update_one_target(target, source, tau)
% SOFT_UPDATE_ONE_TARGET
% target <- tau * source + (1-tau) * target

    target.W1 = tau * source.W1 + (1-tau) * target.W1;
    target.b1 = tau * source.b1 + (1-tau) * target.b1;

    target.W2 = tau * source.W2 + (1-tau) * target.W2;
    target.b2 = tau * source.b2 + (1-tau) * target.b2;
end