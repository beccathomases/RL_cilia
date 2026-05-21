% angvel_from_action: convert actions, tuple of {-1,0,1} into angular velocity using
%  information about the model from the parameter data structure (params)
%
function phidot = angvel_from_action(action,params)
  action = action(:);
  dphi   = params.dphi(:);
  phidot = action .* dphi / params.dt;
end