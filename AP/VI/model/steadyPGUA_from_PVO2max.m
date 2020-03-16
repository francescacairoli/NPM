function PGUA_ss = steadyPGUA_from_PVO2max(PVO2max,params)
    PGUA_ss = params.PGUA_a*PVO2max^2 + params.PGUA_b*PVO2max + params.PGUA_c;
end