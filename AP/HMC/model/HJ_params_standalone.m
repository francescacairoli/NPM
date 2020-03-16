
function p = HJ_params_standalone(BW)

p.BW = BW;
p.ka_int = 0.025;
p.EGP_0 = 0.0158*BW;
p.F_01 = 0.0104*BW;
p.V_G = 0.1797*BW;
p.k12 = 0.0793;
p.R_thr=9;
p.R_cl=0.003;
p.Ag = 0.8121;
p.tMaxg = 48.8385;
p.Ug_ceil = 0.0275*BW;
p.K = 0.7958;
p.kia1 = 0.0113;
p.kia2 = 0.0197;
p.k_e = 0.1735;
p.Vmax_LD = 2.9639;
p.km_LD = 47.5305;
p.ka_1 = 0.007;
p.ka_2 = 0.0331;
p.ka_3 = 0.0308;
p.SIT = 0.0046;
p.SID = 0.0006;
p.SIE = 0.0384;
p.V_I = 0.1443*BW;
p.M_PGU_f = 1/35;
p.M_HGP_f = 1/155;
p.M_PIU_f = 2.4;
p.PGUA_rate = 1/30;
p.PGUA_a = 0.006;
p.PGUA_b = 1.2264;
p.PGUA_c = -10.1952;
p.PVO2max_rate = 5/3;

end

