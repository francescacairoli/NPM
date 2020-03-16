function [ X, T ] = gen_labels( X, tSim)

addpath('model')
% generate parameters
load('model/HJ_params.mat');
u = basal_iir*ones(1,tSim);
dists = repmat(rest_dists,1,tSim);

% Change this particular warning into an error
warnId = 'MATLAB:ode45:IntegrationTolNotMet';
warnstate = warning('error', warnId);

myf = @(t,x)ODE_wrapper(t,x,u,p,dists);


nSamples = size(X,2);
T = zeros(1,nSamples);

i=1;
while i<=nSamples
    
    x0_mod = x0;

    x0_mod(1:length(X(:,i)))=X(:,i)';

    [~,yy] = ode45(myf,[0 tSim],x0_mod);
    T(i) = prod((yy(:,1)/p.V_G>3.9));
    i=i+1;
end
