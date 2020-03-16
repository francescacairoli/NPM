
function [x0,basal_iir] = buildInitState_standalone(target,x,params)
% these are passed as arguments
Q2i_0 = x(1);
I_0 = x(2);


G_0 = target;

% insulin kinetics
Q3_0 = I_0*params.V_I;

Q1a_0 = Q2i_0;

Q1b_0 = (params.k_e*Q3_0 - params.kia1*Q2i_0)./params.kia2;
basal_iir = (params.kia1*Q1a_0 + Q1a_0*params.Vmax_LD/(params.km_LD + Q1a_0) )/params.K;


%effect of insulin
x1_0 = params.SIT*I_0;
x2_0 = params.SID*I_0;
x3_0 = params.SIE*I_0;

%glucose kinetics
Q1_0 = G_0*params.V_G;
Q2_0 = x1_0*Q1_0/(params.k12+x2_0);

G1_0 = 0;
G2_0 = 0;

C_0 = G_0;

restO2 = 8;
%PVO2max and PGUA at rest (8 and 0, resp)
PVO2max_0 = restO2;
PGUA_0 = steadyPGUA_from_PVO2max(PVO2max_0,params);

x0 = [Q1_0 Q2_0 Q1a_0 Q2i_0 Q1b_0 Q3_0 x1_0 x2_0 x3_0 G1_0 G2_0 C_0 PGUA_0 PVO2max_0];

end