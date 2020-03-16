function dydt = ODE_wrapper(t,y,u,params,dists)

if length(u)>1
    u = u(min(end,floor(t+1)));
end

if size(dists,2)>1
    dists=dists(:,min(end,floor(t+1)));
end

dydt = HJ_sys_standalone(y,u,params,dists);

end