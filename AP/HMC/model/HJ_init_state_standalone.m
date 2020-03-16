function [y0,basal_iir,rest_dists] = HJ_init_state_standalone(target_BG,p)
% Given the target glucose level G, we find initial values
% for the basal insulin, I (x(2)), and Q2i (x(1)). 
% All the remaining initial conditions are derived algebraically from I, G,
% and Q2i


rest_dists = [0; 0; 8];

function y = fun_wrapper(x)    
    [initState,b_iir] = buildInitState_standalone(target_BG,x,p);
    y = HJ_sys_standalone(initState,b_iir,p,rest_dists);
end

myf = @(x)fun_wrapper(x);

options_fsolve = optimoptions('fsolve','TolFun', 1e-08, 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
% initial point for search
init_x0 = [20*p.V_I*rand(1,1) 20*rand(1,1)];
x_opt = fsolve(myf,init_x0,options_fsolve);

[y0,basal_iir] = buildInitState_standalone(target_BG,x_opt,p);


end

