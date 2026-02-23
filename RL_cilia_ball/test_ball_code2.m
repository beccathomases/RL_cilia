% parameters for the default 2-ball cilia
%
P = setdefaultparams_ciliaball

% intialize the configuration
%
phi = [0,-P.phimax(2)];

% set of actions
%
A = [repmat([1 1],5,1); repmat([-1 1],5,1);repmat([-1 -1],5,1);repmat([1 -1],5,1)];


% compute the sequence of positions and velocities
%
Nsteps = length(A);
Q = zeros(Nsteps,1);
U = zeros(Nsteps,2);
S = zeros(Nsteps+1,2);
S(1,:) = phi;
for k=1:Nsteps
  [Q(k),S(k+1,:)]=cilia_ball_reward(S(k,:),A(k,:),P);
end

% plotting
%
time = P.dt*(0:Nsteps-1);
set(groot,'defaultLineLineWidth',4)
set(groot,'defaultAxesFontSize',20);


% plot the bending angles
%
figure(1);
plot(time,S(1:Nsteps,:));
xlabel('time');
ylabel('angle');
legend('\phi_{1}','\phi_{2}');

% plot the reward
%
figure(2);
plot(time,Q,'o-','markersize',12);
hold on;
plot(time,mean(Q)*ones(Nsteps,1),'k--');
xlabel('time');
ylabel('flux');
legend('flux','mean flux');
hold off;

% make a movie
%
for k=1:Nsteps
  X=position_from_angle(S(k,:),P);
  phidot = angvel_from_action(A(k,:),P);
  U=velocity_from_angvel(S(k,:),phidot,P);  
  
  XX = [P.X0; X];
  
  
  figure(3);
  plot(XX(:,1),XX(:,3));
  hold on;
  quiver(X(:,1),X(:,3),U(:,1),U(:,3),'r','linewidth',2);
  plot(X(:,1),X(:,3),'r.','markersize',40);
  plot([-1 1],[0 0],'k','linewidth',5);
  hold on;
  xlim([-1 1]);
  ylim([-0.25 1.5]);
  set(gca,'Plotboxaspectratio',[2 1.75 1]);
  hold off;
  pause(0.1);

end

% plot all the shapes
%
figure(4);
colors = hsv(Nsteps);
for k=1:Nsteps
  X=position_from_angle(S(k,:),P);  
  XX = [P.X0; X];
  
    plot(XX(:,1),XX(:,3),'color',colors(k,:));
    hold on;
    plot(X(:,1),X(:,3),'color',colors(k,:),'markersize',40);
    plot([-1 1],[0 0],'k','linewidth',5);
    xlim([-1 1]);
    ylim([-0.25 1.5]);
    set(gca,'Plotboxaspectratio',[2 1.75 1]);
    
end;
hold off;





