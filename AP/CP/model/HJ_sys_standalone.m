% y: state variables -- vector of length 14
% u: control input
% params: parameters
% dists: disturbances at t (vector of length 3)
function dydt = HJ_sys_standalone(y,u,params,dists)

dydt = zeros(length(y),1);

%% extract disturbances
% ingested CHO
D = dists(1);
% active muscular mass at current time
MM = dists(2);
% max oxygen at current time
targetPVo2max = dists(3);

%% extract variables

% glucose kinetics
% masses of glucose in the accessible and non-accessible compartments respectively, in mmol.
Q1 = y(1);
Q2 = y(2);

% Measured glucose concentration
G = Q1/params.V_G;

% corrected non-insulin mediated glucose uptake [Hovorka04]
if(G>=4.5)
    F_01c = params.F_01;
else
    F_01c = params.F_01*G/4.5;
end

if(G>=9)
	F_R = 0.003*(G-9)*params.V_G;
else
	F_R = 0;
end


% insulin kinetics
%insulin mass through the slow absorption pathway,
Q1a = y(3);
Q2i = y(4);
%faster channel for insulin absorption
Q1b = y(5);
%plasma insulin mass
Q3 = y(6);
%plasma insulin concentration
I = Q3./params.V_I;

% insulin dynamics
% x1 (min-1), x2 (min-1) and x3 (unitless) represent 
% the effect of insulin on glucose distribution, 
% glucose disposal and suppression of endogenous glucose 
% production, respectively
x1 = y(7);
x2 = y(8);
x3 = y(9);

k_b1 = params.ka_1*params.SIT;
k_b2 = params.ka_2*params.SID;
k_b3 = params.ka_3*params.SIE;

% Subsystem of glucose absorption from gut
% Glucose masses in the accessible and nonaccessible compartments
G1 = y(10);
G2 = y(11);

tmax = max(params.tMaxg,G2/params.Ug_ceil);
U_g = G2/tmax;

% interstitial glucose
C = y(12);


% exercise 
PGUA = y(13);
PVO2max = y(14);
M_PGU = 1 + PGUA*MM*params.M_PGU_f;
M_PIU = 1 + MM*params.M_PIU_f;
M_HGP = 1 + PGUA*MM*params.M_HGP_f;
%PGUA_ss = p.PGUA_a*PVO2max^2 + p.PGUA_b*PVO2max + p.PGUA_c;
PGUA_ss = steadyPGUA_from_PVO2max(PVO2max,params);

%% compute change rates
% use flow variables to avoid duplicated computation

%% Glucose kinetics
Q1_to_Q2_flow = x1*Q1 - params.k12*Q2;
Q1dt = -F_01c -Q1_to_Q2_flow - F_R + U_g +  params.EGP_0*(1 - x3);
Q2dt = Q1_to_Q2_flow -x2*Q2;
dydt(1) = Q1dt;
dydt(2) = Q2dt;

%% insulin kinetics
Q1a_to_Q2i_flow = params.kia1*Q1a;
Q2i_to_Q3_flow = params.kia1*Q2i;
Q1b_to_Q3_flow = params.kia2*Q1b;
insulin_ratio = params.K*u;

Q1adt = insulin_ratio - Q1a_to_Q2i_flow - params.Vmax_LD*Q1a./(params.km_LD+Q1a);
Q2idt = Q1a_to_Q2i_flow - Q2i_to_Q3_flow;
Q1bdt = u - insulin_ratio - Q1b_to_Q3_flow - params.Vmax_LD*Q1b./(params.km_LD+Q1b);
Q3dt = Q2i_to_Q3_flow + Q1b_to_Q3_flow - params.k_e*Q3;

dydt(3)=Q1adt;
dydt(4)=Q2idt;
dydt(5)=Q1bdt;
dydt(6)=Q3dt;

%% insulin dynamics
x1dt = -params.ka_1*x1 + M_PGU*M_PIU*k_b1*I;
x2dt = -params.ka_2*x2 + M_PGU*M_PIU*k_b2*I;
x3dt = -params.ka_3*x3 + M_HGP*k_b3*I;
dydt(7) = x1dt;
dydt(8) = x2dt;
dydt(9) = x3dt;


%% Glucose absorption from gut
G1_to_G2_flow = G1./tmax;
G1dt =  - G1_to_G2_flow + params.Ag*D;
G2dt =  G1_to_G2_flow - G2./tmax;
dydt(10) = G1dt;
dydt(11) = G2dt;


%% interstitial glucose
Cdt = params.ka_int*(G-C);
dydt(12) = Cdt;


%% exercise
PGUAdt = -params.PGUA_rate*PGUA +params.PGUA_rate*PGUA_ss;
dydt(13) = PGUAdt;

PVO2maxdt = -params.PVO2max_rate*PVO2max +params.PVO2max_rate*targetPVo2max;
dydt(14) = PVO2maxdt;

end