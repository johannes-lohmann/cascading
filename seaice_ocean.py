import numpy as np
import matplotlib.pyplot as pl
import math
import time
from scipy.interpolate import UnivariateSpline
from scipy import signal
from scipy.optimize import fmin
from scipy.ndimage import filters
import statsmodels.api as sm
import seaice_ocean_cy_noise as ode_cy
from ews_functions import detrend_cubic, smoothing_filter_gauss, detrend_spline, butterworth_zerophase, DFA, sinc_highpass, estimated_ac1_alt2, ac_fit_exp, DFA, DCCA
import ews_cy as ews
from ts_analysis import estimated_autocorrelation_cut as ac
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kendalltau as ktau
from  matplotlib import colors as clrs
from scipy.signal import savgol_filter
import matplotlib as mpl
import warnings

pl.rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size'   : 16})
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['axes.xmargin'] = 0.03
mpl.rcParams['axes.ymargin'] = 0.03
mpl.rcParams['axes.unicode_minus'] = False


prop_cycle = pl.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def MAIN():
        start_time1 = time.time()
        
        h = 0.0005; loop = 150.
        T = 1500 #simulation time in years
        N_T = int(round(float(T)/h/loop)); t = np.linspace(0, T, N_T)

        delt = 0.43 #albedo difference ocean - sea ice
        h_alph = 0.5
        F = 1/28. #ocean forcing on sea ice
        R0 = -0.1 #sea ice export in %
        B = 0.45 #outgoing longwave radiation coeff
        Lm = 1.25 #incoming longwave
        R=0.0 #sea ice import bifurcation parameter

        eta1 = 3.0; eta2 = 1.; eta3 = 0.3
        Icov = 1.1715#1.445##fixed point for h=0.5: 1.156
        fact = 0.35 #determine size of jump in eta1
        B0 = 1./Icov*fact
        RT = 1/200. #inverse timescale of ocean model
        sigT = 0.07
        sigS = 0.07

        start = 200
        dur = 1000
        ampl = 0.29
        ampl2 = 0.15
        wait = 200#
        parameters = np.asarray([h, loop, T, N_T, delt, h_alph, F, R0, B, Lm, R, eta1, eta2, eta3, Icov, fact, B0, RT, sigT, sigS, start, dur, ampl, ampl2, wait])
        
        #test_dcca(parameters)
        #test_filter(parameters)
        #ews_variance(parameters)

        #early_warning()
        #early_warning_significance(100,1,0.02,100)
        #early_warning_rate(100,1,0.02,100)
        #stack_grids()
        early_warning_rate_eval()
        
        #simulate()

        print('simulation time (s)', time.time()-start_time1)
        pl.show()

def test_dcca(parameters):
        h, loop, T, N_T, delt, h_alph, F, R0, B, Lm, R, eta1, eta2, eta3, Icov, fact, B0, RT, sigT, sigS, start, dur, ampl, ampl2, wait = parameters
        T = 20000 #simulation time in years
        N_T = int(round(float(T)/h/loop)); t = np.linspace(0, T, N_T)
        sigI = 0.02; sigT = 0.02; sigS = 0.02
        t_kernel_g_I = 50; t_kernel_g_T = 50

        ampl = 0.25
        ramp = np.asarray(int(round(float(T)/h)+loop)*[-ampl])
        params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT,sigI,sigT,sigS])
        U_0 = [0.573,2.4,2.5]#[1.1563,2.4,2.5]#
        I,Te,S = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)

        I_gauss = I-smoothing_filter_gauss(I, int(t_kernel_g_I/0.075), int(t_kernel_g_I/0.075)*4)
        T_gauss = Te-smoothing_filter_gauss(Te, int(t_kernel_g_T/0.075), int(t_kernel_g_T/0.075)*4)

        dcca = DCCA(I_gauss[int(1000/h/loop):-int(100/h/loop)],T_gauss[int(1000/h/loop):-int(100/h/loop)])

        scale_lim=[4,8]; scale_dens=0.25
        print(np.arange(scale_lim[0], scale_lim[1], scale_dens))
        scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
        print(scales)

def ews_variance(parameters):

        h, loop, T, N_T, delt, h_alph, F, R0, B, Lm, R, eta1, eta2, eta3, Icov, fact, B0, RT, sigT, sigS, start, dur, ampl, ampl2, wait = parameters
        T = 30000 #simulation time in years
        N_T = int(round(float(T)/h/loop)); t = np.linspace(0, T, N_T)
        t = np.linspace(0, T, N_T)
        sigI = 0.01
        ampl = 0.25
        ramp = np.asarray(int(round(float(T)/h)+loop)*[-ampl])
        params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT,sigI,sigT,sigS])
        U_0 = [0.573,2.4,2.5]#[1.1563,2.4,2.5]#

        ### Test variance of variance estimate, using 
        ### the correlation time for the effective number of ind. samples

        samples = 2000
        T_vals = [30,40,50,70,100,130,170,200,250,400,600,800,1000]
        lag_step = 2; lag_stop = 100; lags = range(lag_step, lag_stop, lag_step)
        real = []
        for j in range(len(T_vals)):
                T=T_vals[j]
                N_T = int(round(float(T)/h/loop))
                ramp = np.asarray(int(round(float(T)/h)+loop)*[-ampl])

                var_all = np.empty(samples); corr_all = np.empty(samples); ar_all = np.empty(samples); ac_tau_all = np.empty(samples)
                for i in range(samples):
                        I,Te,S = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
                        #var = ews.cy_var(I)
                        #var = ews.estimated_ac1(I,8)#lag=8
                        var = DFA(I)
                        var_all[i] =  var
                        #ac_vals = [ews.estimated_ac1(I,x) for x in lags]
                        #corr = (1 + 2.*np.sum(ac_vals[1:]))*0.075*lag_step
                        #corr_all[i] = corr
                        #ar = estimated_ac1_alt2(I)
                        #ar_all[i] = ar
                        #ac_tau = ac_fit_exp(np.asarray(lags), ac_vals)*0.075*lag_step
                        #ac_tau_all[i] = ac_tau

                print('Sample mean of DFA exp.', np.mean(var_all))
                print('Sample stand. dev. of DFA exp.', np.std(var_all))

                #print('Sample mean of sum Tcorr', np.mean(corr_all))
                #print('Sample stand. dev. of sum Tcorr', np.std(corr_all))

                #print('Sample mean of AR(1) coeff', np.mean(ar_all))
                #print('Sample stand. dev. of AR(1) coeff', np.std(ar_all))

                #print('Sample mean of tau from exp. AC', np.mean(ac_tau_all))
                #print('Sample stand. dev. of tau from exp. AC', np.std(ac_tau_all))

                real.append(np.std(var_all))
        '''
        pred_curve = [np.sqrt(1./(x/Tcorr)*(mu4-(x/Tcorr-3.)/(x/Tcorr-1.)*mu2**2)) for x in np.linspace(30,1000,100)]

        fig=pl.figure()
        pl.suptitle('St. Dev. of var. est. in EIS12 model (R=0) as func. of data window length. \n Correlation time %s years'%round(Tcorr,1))
        pl.plot(np.linspace(30,1000,100),pred_curve,label='Theory')
        pl.plot(T_vals, real, 'x',label='Simulation')
        pl.legend(loc='best');pl.ylabel('Std (var. estimate)'); pl.xlabel('Window length (years)')
        '''
def test_filter(parameters):

        h, loop, T, N_T, delt, h_alph, F, R0, B, Lm, R, eta1, eta2, eta3, Icov, fact, B0, RT, sigT, sigS, start, dur, ampl, ampl2, wait = parameters
        t = np.linspace(0, T, N_T)
        sigI = 0.01
        cut = 15000; t_kernel_g_I = 100; t_kernel_g_T = 10
        ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,-ampl,int(dur/h)),int((T-start-dur)/h)*[-ampl]),)
        params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT,sigI,sigT,sigS])
        U_0 = [0.573,2.4,2.5]
        I,Te,S = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)

        idx = np.argmax(I<0.0)
        I_align = I[idx-cut:idx]
        t_align = np.linspace(0,cut*h*loop,cut)

        I_gauss_low = smoothing_filter_gauss(I_align, int(t_kernel_g_I/0.075), int(t_kernel_g_I/0.075)*4)#std, window size
        I_gauss = I_align-I_gauss_low
        I_butter_low = butterworth_zerophase(I_align, 0.01)
        I_butter = I_align-I_butter_low

        T_gauss_low = smoothing_filter_gauss(Te, int(t_kernel_g_T/0.075), int(t_kernel_g_T/0.075)*4)#std, window size
        T_gauss = Te-T_gauss_low
        T_butter_low = butterworth_zerophase(Te, 0.01)
        T_butter = Te-T_butter_low

        samples = 100
        filt_vals = [10,12,15,20,25,30,35,40,50,60,70,80,100]

        T=1000
        N_T = int(round(float(T)/h/loop))
        ampl = 0.25;ramp = np.asarray(int(round(float(T)/h)+loop)*[-ampl])

        for j in range(len(filt_vals)):
                t_kernel_g_I = filt_vals[j]
                variances = []
                for i in range(samples):
                        I,Te,S = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
                        I_gauss = I-smoothing_filter_gauss(I, int(t_kernel_g_I/0.075), int(t_kernel_g_I/0.075)*4)
                        variances.append(ews.cy_var(I_gauss))
                print(np.mean(variances))
                        

        '''
        fig=pl.figure()
        pl.subplot(411)
        pl.plot(t_align, I_align)
        pl.plot(t_align, I_gauss_low)
        #pl.plot(t_align, I_butter_low)
        pl.subplot(412)
        pl.plot(t_align, I_gauss)
        #pl.plot(t_align, I_butter)

        pl.subplot(413)
        pl.plot(t, Te)
        pl.plot(t, T_gauss_low)
        #pl.plot(t, T_butter_low)
        pl.subplot(414)
        pl.plot(t, T_gauss)
        #pl.plot(t, T_butter)
        '''


def early_warning_rate(wind, downs, sigI, count):

	delt = 0.43; h_alph = 0.5; F = 1/28.; R0 = -0.1; B = 0.45; Lm = 1.25; R=0.0 

	eta1 = 3.0; eta2 = 1.; eta3 = 0.3
	Icov = 1.156 #fixed point for h=0.5: 1.156
	fact = 0.35; B0 = 1./Icov*fact; RT = 1/200. #inverse timescale of ocean model
	sigT = 0.02; sigS = 0.02

	U_0 = [1.156,2.4,2.5]

	h = 0.0005; loop = 150.
	T = 850
	N_T = int(round(float(T)/h/loop)); t = np.linspace(0, T, N_T)

	dur = 340; start = dur
	ampl = 0.29

	ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,-ampl,int(dur/h)),int((T-start-dur)/h+loop)*[-ampl]),)

	params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT,sigI,sigT,sigS])

	thin = 50

	window = int(wind/(h*loop)) #running window length in output sample timesteps
	samples = 3#20 #3#10
	t_kernel_g_I = 50; t_kernel_g_T = 50

	dur_vals = np.linspace(300,500,2)

	rtip_prob = np.empty(len(dur_vals))

	ac1I_tau_med = np.empty(len(dur_vals)); ac1T_tau_med = np.empty(len(dur_vals))
	varI_tau_med = np.empty(len(dur_vals)); varT_tau_med = np.empty(len(dur_vals))
	dcca_tau_med = np.empty(len(dur_vals)); cc_tau_med = np.empty(len(dur_vals))

	ac1I_tau_medS = np.empty(len(dur_vals)); ac1T_tau_medS = np.empty(len(dur_vals))
	varI_tau_medS = np.empty(len(dur_vals)); varT_tau_medS = np.empty(len(dur_vals))
	dcca_tau_medS = np.empty(len(dur_vals)); cc_tau_medS = np.empty(len(dur_vals))

	for j in range(len(dur_vals)):

		dur = dur_vals[j]; start = dur
		ampl = 0.29
		T = dur+start+2000
		N_T = int(round(float(T)/h/loop)); t = np.linspace(0, T, N_T)
		ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,-ampl,int(dur/h)),int((T-start-dur)/h+loop)*[-ampl]),)

		ac1I_all=[];ac1T_all=[];varI_all=[];varT_all=[];dcca_all=[];cc_all=[];tip_times=[]
		rtip_count=0
		for i in range(samples):

		        I0,Te0,S0 = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)

		        idx=len(I0); idx = np.argmax(I0<0.0)
		        tip_times.append(idx*h*loop)

		        I_align = I0
		        T_align = Te0
		 
		        I_gauss = I_align-smoothing_filter_gauss(I_align, int(t_kernel_g_I/0.075), int(t_kernel_g_I/0.075)*4)
		        T_gauss = T_align-smoothing_filter_gauss(T_align, int(t_kernel_g_T/0.075), int(t_kernel_g_T/0.075)*4)

		        acI,varI,acT,varT,dcca,cc = calc_ews_sign(I_gauss,T_gauss,int(window),thin)
		        ac1I_all.append(acI); varI_all.append(varI)
		        ac1T_all.append(acT); varT_all.append(varT)
		        dcca_all.append(dcca); cc_all.append(cc)

		        if Te0[-1]>S0[-1]:
		                rtip_count += 1

		print('fraction of R-tippings :' , rtip_count/float(samples))
		rtip_prob[j] = rtip_count/float(samples)
		print('average, sdev, min, max time of I tipping: ', np.mean(tip_times),np.std(tip_times),min(tip_times),max(tip_times))
		ews_cut_time = np.mean(tip_times)-t_kernel_g_I

		cut_idx_ews = int((ews_cut_time-wind)/h/loop/thin)
		start_idx_ews = int((start-wind)/h/loop/thin)

		mean_ac1I,percentile_5_ac1I,percentile_95_ac1I = calc_mean_conf(ac1I_all,cut_idx_ews,samples)
		mean_ac1T,percentile_5_ac1T,percentile_95_ac1T = calc_mean_conf(ac1T_all,cut_idx_ews,samples)
		mean_varI,percentile_5_varI,percentile_95_varI = calc_mean_conf(varI_all,cut_idx_ews,samples)
		mean_varT,percentile_5_varT,percentile_95_varT = calc_mean_conf(varT_all,cut_idx_ews,samples)
		mean_dcca,percentile_5_dcca,percentile_95_dcca = calc_mean_conf(dcca_all,cut_idx_ews,samples)
		mean_cc,percentile_5_cc,percentile_95_cc = calc_mean_conf(cc_all,cut_idx_ews,samples)

		mean_95_ac1I = np.mean(percentile_95_ac1I[:start_idx_ews])
		mean_95_ac1T = np.mean(percentile_95_ac1T[:start_idx_ews])
		mean_95_varI = np.mean(percentile_95_varI[:start_idx_ews])
		mean_95_varT = np.mean(percentile_95_varT[:start_idx_ews])
		mean_95_dcca = np.mean(percentile_95_dcca[:start_idx_ews])
		mean_95_cc = np.mean(percentile_95_cc[:start_idx_ews])
		

		tau0_ac1I = [ktau(range(len(ac1I_all[i][:start_idx_ews])),ac1I_all[i][:start_idx_ews])[0] for i in range(samples)]
		tau_ac1I = [ktau(range(len(ac1I_all[i][start_idx_ews:cut_idx_ews])),ac1I_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		tau0_ac1T = [ktau(range(len(ac1T_all[i][:start_idx_ews])),ac1T_all[i][:start_idx_ews])[0] for i in range(samples)]
		tau_ac1T = [ktau(range(len(ac1T_all[i][start_idx_ews:cut_idx_ews])),ac1T_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		tau0_varI = [ktau(range(len(varI_all[i][:start_idx_ews])),varI_all[i][:start_idx_ews])[0] for i in range(samples)]
		tau_varI = [ktau(range(len(varI_all[i][start_idx_ews:cut_idx_ews])),varI_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		tau0_varT = [ktau(range(len(varT_all[i][:start_idx_ews])),varT_all[i][:start_idx_ews])[0] for i in range(samples)]
		tau_varT = [ktau(range(len(varT_all[i][start_idx_ews:cut_idx_ews])),varT_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		tau0_dcca = [ktau(range(len(dcca_all[i][:start_idx_ews])),dcca_all[i][:start_idx_ews])[0] for i in range(samples)]
		tau_dcca = [ktau(range(len(dcca_all[i][start_idx_ews:cut_idx_ews])),dcca_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		tau0_cc = [ktau(range(len(cc_all[i][:start_idx_ews])),cc_all[i][:start_idx_ews])[0] for i in range(samples)]
		tau_cc = [ktau(range(len(cc_all[i][start_idx_ews:cut_idx_ews])),cc_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		ac1I_tau_med[j] = np.median(tau_ac1I)
		ac1T_tau_med[j] = np.median(tau_ac1T)
		varI_tau_med[j] = np.median(tau_varI)
		varT_tau_med[j] = np.median(tau_varT)
		dcca_tau_med[j] = np.median(tau_dcca)
		cc_tau_med[j] = np.median(tau_cc)

		ac1I_tau_medS[j] = comp_sign(np.percentile(tau_ac1I,5.), np.percentile(tau0_ac1I,95.))
		ac1T_tau_medS[j] = comp_sign(np.percentile(tau_ac1T,5.), np.percentile(tau0_ac1T,95.))
		varI_tau_medS[j] = comp_sign(np.percentile(tau_varI,5.), np.percentile(tau0_varI,95.))
		varT_tau_medS[j] = comp_sign(np.percentile(tau_varT,5.), np.percentile(tau0_varT,95.))
		dcca_tau_medS[j] = comp_sign(np.percentile(tau_dcca,5.), np.percentile(tau0_dcca,95.))
		cc_tau_medS[j] = comp_sign(np.percentile(tau_cc,5.), np.percentile(tau0_cc,95.))

		print(j)
		print(np.median(tau_ac1I), np.percentile(tau_ac1I,5.), np.percentile(tau0_ac1I,95.))
		print(np.median(tau_ac1T), np.percentile(tau_ac1T,5.), np.percentile(tau0_ac1T,95.))
		print(np.median(tau_varI), np.percentile(tau_varI,5.), np.percentile(tau0_varI,95.))
		print(np.median(tau_varT), np.percentile(tau_varT,5.), np.percentile(tau0_varT,95.))
		print(np.median(tau_dcca), np.percentile(tau_dcca,5.), np.percentile(tau0_dcca,95.))
		print(np.median(tau_cc), np.percentile(tau_cc,5.), np.percentile(tau0_cc,95.))
		print('-------------------------------')

	to_file = np.asarray([rtip_prob,ac1I_tau_med,ac1T_tau_med,varI_tau_med,varT_tau_med,dcca_tau_med,cc_tau_med])
	np.save('median_data',to_file)
	to_file = np.asarray([ac1I_tau_medS,ac1T_tau_medS,varI_tau_medS,varT_tau_medS,dcca_tau_medS,cc_tau_medS])
	np.save('significance_data',to_file)



	fig=pl.figure()
	pl.plot(dur_vals, rtip_prob)


def early_warning_rate_eval():

        rtip_prob,ac1I_tau_med,ac1T_tau_med,varI_tau_med,varT_tau_med,dcca_tau_med,cc_tau_med = np.load('median_data_sigI002.npy')
        ac1I_tau_medS,ac1T_tau_medS,varI_tau_medS,varT_tau_medS,dcca_tau_medS,cc_tau_medS = np.load('significance_data_sigI002.npy')

        rtip_prob2,ac1I_tau_med2,ac1T_tau_med2,varI_tau_med2,varT_tau_med2,dcca_tau_med2,cc_tau_med2 = np.load('median_data_sigI02.npy')
        rtip_prob05,ac1I_tau_med05,ac1T_tau_med05,varI_tau_med05,varT_tau_med05,dcca_tau_med05,cc_tau_med05 = np.load('median_data_sigI0005.npy')

        rtip_prob4,ac1I_tau_med4,ac1T_tau_med4,varI_tau_med4,varT_tau_med4,dcca_tau_med4,cc_tau_med4 = np.load('median_data_sigI004.npy')
        ac1I_tau_medS4,ac1T_tau_medS4,varI_tau_medS4,varT_tau_medS4,dcca_tau_medS4,cc_tau_medS4 = np.load('significance_data_sigI004.npy')

        rtip_prob1,ac1I_tau_med1,ac1T_tau_med1,varI_tau_med1,varT_tau_med1,dcca_tau_med1,cc_tau_med1 = np.load('median_data_sigI001.npy')
        ac1I_tau_medS1,ac1T_tau_medS1,varI_tau_medS1,varT_tau_medS1,dcca_tau_medS1,cc_tau_medS1 = np.load('significance_data_sigI001.npy')

        rtip_prob8,ac1I_tau_med8,ac1T_tau_med8,varI_tau_med8,varT_tau_med8,dcca_tau_med8,cc_tau_med8 = np.load('median_data_sigI008.npy')
        ac1I_tau_medS8,ac1T_tau_medS8,varI_tau_medS8,varT_tau_medS8,dcca_tau_medS8,cc_tau_medS8 = np.load('significance_data_sigI008.npy')

        rtip_probT4,ac1I_tau_medT4,ac1T_tau_medT4,varI_tau_medT4,varT_tau_medT4,dcca_tau_medT4,cc_tau_medT4 = np.load('median_data_sigT004.npy')
        rtip_probT1,ac1I_tau_medT1,ac1T_tau_medT1,varI_tau_medT1,varT_tau_medT1,dcca_tau_medT1,cc_tau_medT1 = np.load('median_data_sigT001.npy')
        rtip_probT05,ac1I_tau_medT05,ac1T_tau_medT05,varI_tau_medT05,varT_tau_medT05,dcca_tau_medT05,cc_tau_medT05 = np.load('median_data_sigT0005.npy')
        rtip_probT11,ac1I_tau_medT11,ac1T_tau_medT11,varI_tau_medT11,varT_tau_medT11,dcca_tau_medT11,cc_tau_medT11 = np.load('median_data_sigT01.npy')

        print(rtip_probT4)
        print(rtip_probT1)

        print(ac1I_tau_medT4)
        print(ac1I_tau_medT1)

        dur_vals = np.linspace(280,600,30)

        t = np.linspace(280,600,300)
        spl = UnivariateSpline(dur_vals, rtip_prob1, s=0, k=2)
        yhat = spl(t)


        fig=pl.figure(figsize=(5.,4.))
        pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
        pl.plot(100*[408.], np.linspace(0.,1,100), ':', color='tomato')
        pl.plot(dur_vals, savgol_filter(rtip_prob05, 5, 3),alpha=.25,color='black',label='$\sigma_I$ = 0.005')
        pl.plot(dur_vals, savgol_filter(rtip_prob1, 5, 3),alpha=.35,color='black',label='$\sigma_I$ = 0.01')
        pl.plot(dur_vals, savgol_filter(rtip_prob, 5, 3),alpha=.5,color='black',label='$\sigma_I$ = 0.02')
        pl.plot(dur_vals, savgol_filter(rtip_prob4, 5, 3),alpha=.7,color='black',label='$\sigma_I$ = 0.04')
        pl.plot(dur_vals, savgol_filter(rtip_prob8, 5, 3),alpha=.85,color='black',label='$\sigma_I$ = 0.08')
        pl.plot(dur_vals, savgol_filter(rtip_prob2, 5, 3),color='black',label='$\sigma_I$ = 0.2')

        pl.xlabel('Ramping duration (years)'); pl.ylabel('Tipping probability')

        fig=pl.figure(figsize=(5.,4.))
        pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
        pl.plot(100*[408.], np.linspace(0.,1,100), ':', color='tomato')
        pl.plot(dur_vals, savgol_filter(rtip_probT05, 5, 3),alpha=.25,color='black',label='$\sigma_T$ = 0.005')
        pl.plot(dur_vals, savgol_filter(rtip_probT1, 5, 3),alpha=.4,color='black',label='$\sigma_T$ = 0.01')
        pl.plot(dur_vals, savgol_filter(rtip_prob, 5, 3),alpha=.65,color='black',label='$\sigma_T$ = 0.02')
        pl.plot(dur_vals, savgol_filter(rtip_probT4, 5, 3),alpha=.85,color='black',label='$\sigma_T$ = 0.04')
        pl.plot(dur_vals, savgol_filter(rtip_probT11, 5, 3), color='black',label='$\sigma_T$ = 0.1')

        pl.xlabel('Ramping duration (years)'); pl.ylabel('Tipping probability')

        '''

        fig=pl.figure()
        pl.suptitle('AR(1) I')
        pl.plot(dur_vals, ac1I_tau_med1,label='$\sigma_I$ = 0.01')
        pl.plot(dur_vals, ac1I_tau_med,linewidth=2.,label='$\sigma_I$ = 0.02')
        pl.plot(dur_vals, ac1I_tau_med4,label='$\sigma_I$ = 0.04')
        pl.plot(dur_vals, ac1I_tau_med8,label='$\sigma_I$ = 0.08')
        pl.plot(dur_vals, ac1I_tau_medT1,'--',label='$\sigma_T$ = 0.01')
        pl.plot(dur_vals, ac1I_tau_medT4,'--',label='$\sigma_T$ = 0.04')


        fig=pl.figure()
        pl.suptitle('Variance I')
        pl.plot(dur_vals, varI_tau_med1,label='$\sigma_I$ = 0.01')
        pl.plot(dur_vals, varI_tau_med,linewidth=2.,label='$\sigma_I$ = 0.02')
        pl.plot(dur_vals, varI_tau_med4,label='$\sigma_I$ = 0.04')
        pl.plot(dur_vals, varI_tau_med8,label='$\sigma_I$ = 0.08')
        pl.plot(dur_vals, varI_tau_medT1,'--',label='$\sigma_T$ = 0.01')
        pl.plot(dur_vals, varI_tau_medT4,'--',label='$\sigma_T$ = 0.04')

        '''

        fig=pl.figure()
        pl.suptitle('AR(1) T')
        pl.plot(dur_vals, savgol_filter(ac1T_tau_med05,11,2) ,linewidth=.25,color='black' ,label='$\sigma_I$ = 0.005')
        pl.plot(dur_vals, savgol_filter(ac1T_tau_med05,11,2) ,linewidth=.5,color='black',label='$\sigma_I$ = 0.005')
        pl.plot(dur_vals, savgol_filter(ac1T_tau_med1,11,2),linewidth=1.,color='black',label='$\sigma_I$ = 0.01')
        pl.plot(dur_vals, savgol_filter(ac1T_tau_med,11,2),linewidth=1.5,color='black',label='$\sigma_I$ = 0.02')
        pl.plot(dur_vals, savgol_filter(ac1T_tau_med4,11,2),linewidth=2.,color='black',label='$\sigma_I$ = 0.04')
        pl.plot(dur_vals, savgol_filter(ac1T_tau_med8,11,2),linewidth=2.5,color='black',label='$\sigma_I$ = 0.08')
        pl.plot(dur_vals, savgol_filter(ac1T_tau_med2,11,2),linewidth=2.,label='$\sigma_I$ = 0.2')
        pl.legend(loc='best')

        fig=pl.figure()
        pl.suptitle('Variance T')
        pl.plot(dur_vals, savgol_filter(varT_tau_med05,11,2),linewidth=.25,color='black' ,label='$\sigma_I$ = 0.005')
        pl.plot(dur_vals, savgol_filter(varT_tau_med1,11,2),linewidth=.5,color='black',label='$\sigma_I$ = 0.01')
        pl.plot(dur_vals, savgol_filter(varT_tau_med,11,2),linewidth=1.5,color='black',label='$\sigma_I$ = 0.02')
        pl.plot(dur_vals, savgol_filter(varT_tau_med4,11,2),linewidth=2.,color='black',label='$\sigma_I$ = 0.04')
        pl.plot(dur_vals, savgol_filter(varT_tau_med8,11,2),linewidth=2.5,color='black',label='$\sigma_I$ = 0.08')
        pl.plot(dur_vals, savgol_filter(varT_tau_med2,11,2),linewidth=2.,label='$\sigma_I$ = 0.2')
        pl.legend(loc='best')

        fig=pl.figure()
        pl.suptitle('Cross-correlation')
        pl.plot(dur_vals, cc_tau_med1,label='$\sigma_I$ = 0.01')
        pl.plot(dur_vals, cc_tau_med,linewidth=2.,label='$\sigma_I$ = 0.02')
        pl.plot(dur_vals, cc_tau_med4,label='$\sigma_I$ = 0.04')
        pl.plot(dur_vals, cc_tau_med8,label='$\sigma_I$ = 0.08')

        pl.plot(dur_vals, cc_tau_medT1,'--',label='$\sigma_T$ = 0.01')
        pl.plot(dur_vals, cc_tau_medT4,'--',label='$\sigma_T$ = 0.04')
        

def early_warning_significance(wind, downs, sigI, count):

	delt = 0.43; h_alph = 0.5; F = 1/28.; R0 = -0.1; B = 0.45; Lm = 1.25; R=0.0 

	eta1 = 3.0; eta2 = 1.; eta3 = 0.3
	Icov = 1.156 #fixed point for h=0.5: 1.156
	fact = 0.35; B0 = 1./Icov*fact; RT = 1/200. #inverse timescale of ocean model
	sigT = 0.02; sigS = 0.02

	U_0 = [1.156,2.4,2.5]

	h = 0.0005; loop = 150.
	T = 850
	N_T = int(round(float(T)/h/loop)); t = np.linspace(0, T, N_T)

	dur = 340; start = dur
	ampl = 0.29

	ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,-ampl,int(dur/h)),int((T-start-dur)/h+loop)*[-ampl]),)

	params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT,sigI,sigT,sigS])

	thin = 50

	window = int(wind/(h*loop)) #running window length in output sample timesteps
	samples = 40
	t_kernel_g_I = 50; t_kernel_g_T = 50

	wind_vals = np.linspace(50,350,3)
	filt = np.linspace(30,100,4)

	ac1I_tau_med = np.empty((len(wind_vals),len(filt))); ac1T_tau_med = np.empty((len(wind_vals),len(filt)))
	varI_tau_med = np.empty((len(wind_vals),len(filt))); varT_tau_med = np.empty((len(wind_vals),len(filt)))
	dcca_tau_med = np.empty((len(wind_vals),len(filt))); cc_tau_med = np.empty((len(wind_vals),len(filt)))

	ac1I_tau_medS = np.empty((len(wind_vals),len(filt))); ac1T_tau_medS = np.empty((len(wind_vals),len(filt)))
	varI_tau_medS = np.empty((len(wind_vals),len(filt))); varT_tau_medS = np.empty((len(wind_vals),len(filt)))
	dcca_tau_medS = np.empty((len(wind_vals),len(filt))); cc_tau_medS = np.empty((len(wind_vals),len(filt)))

	for k in range(len(wind_vals)):
		for j in range(len(filt)):
		        wind = wind_vals[k]
		        window = int(wind/(h*loop))
		        t_kernel_g_I = filt[j]; t_kernel_g_T = filt[j]

		        ac1I_all=[];ac1T_all=[];varI_all=[];varT_all=[];dcca_all=[];cc_all=[];tip_times=[]
		        rtip_count=0
		        for i in range(samples):

		                I0,Te0,S0 = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)

		                idx=len(I0); idx = np.argmax(I0<0.0)
		                tip_times.append(idx*h*loop)

		                I_align = I0
		                T_align = Te0
		         
		                I_gauss = I_align-smoothing_filter_gauss(I_align, int(t_kernel_g_I/0.075), int(t_kernel_g_I/0.075)*4)
		                T_gauss = T_align-smoothing_filter_gauss(T_align, int(t_kernel_g_T/0.075), int(t_kernel_g_T/0.075)*4)

		                acI,varI,acT,varT,dcca,cc = calc_ews_sign(I_gauss,T_gauss,int(window),thin)
		                ac1I_all.append(acI); varI_all.append(varI)
		                ac1T_all.append(acT); varT_all.append(varT)
		                dcca_all.append(dcca); cc_all.append(cc)

		                #if Te0[-1]>S0[-1]:
		                #        rtip_count += 1

		        #print('fraction of R-tippings :' , rtip_count/float(samples))
		        print('average, sdev, min, max time of I tipping: ', np.mean(tip_times),np.std(tip_times),min(tip_times),max(tip_times))
		        ews_cut_time = np.mean(tip_times)-t_kernel_g_I

		        cut_idx_ews = int((ews_cut_time-wind)/h/loop/thin)
		        start_idx_ews = int((start-wind)/h/loop/thin)

		        mean_ac1I,percentile_5_ac1I,percentile_95_ac1I = calc_mean_conf(ac1I_all,cut_idx_ews,samples)
		        mean_ac1T,percentile_5_ac1T,percentile_95_ac1T = calc_mean_conf(ac1T_all,cut_idx_ews,samples)
		        mean_varI,percentile_5_varI,percentile_95_varI = calc_mean_conf(varI_all,cut_idx_ews,samples)
		        mean_varT,percentile_5_varT,percentile_95_varT = calc_mean_conf(varT_all,cut_idx_ews,samples)
		        mean_dcca,percentile_5_dcca,percentile_95_dcca = calc_mean_conf(dcca_all,cut_idx_ews,samples)
		        mean_cc,percentile_5_cc,percentile_95_cc = calc_mean_conf(cc_all,cut_idx_ews,samples)

		        mean_95_ac1I = np.mean(percentile_95_ac1I[:start_idx_ews])
		        mean_95_ac1T = np.mean(percentile_95_ac1T[:start_idx_ews])
		        mean_95_varI = np.mean(percentile_95_varI[:start_idx_ews])
		        mean_95_varT = np.mean(percentile_95_varT[:start_idx_ews])
		        mean_95_dcca = np.mean(percentile_95_dcca[:start_idx_ews])
		        mean_95_cc = np.mean(percentile_95_cc[:start_idx_ews])

		        
		        print(len(ac1I_all[0][:start_idx_ews]), len(ac1I_all[0][start_idx_ews:cut_idx_ews]))

		        tau0_ac1I = [ktau(range(len(ac1I_all[i][:start_idx_ews])),ac1I_all[i][:start_idx_ews])[0] for i in range(samples)]
		        tau_ac1I = [ktau(range(len(ac1I_all[i][start_idx_ews:cut_idx_ews])),ac1I_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		        tau0_ac1T = [ktau(range(len(ac1T_all[i][:start_idx_ews])),ac1T_all[i][:start_idx_ews])[0] for i in range(samples)]
		        tau_ac1T = [ktau(range(len(ac1T_all[i][start_idx_ews:cut_idx_ews])),ac1T_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		        tau0_varI = [ktau(range(len(varI_all[i][:start_idx_ews])),varI_all[i][:start_idx_ews])[0] for i in range(samples)]
		        tau_varI = [ktau(range(len(varI_all[i][start_idx_ews:cut_idx_ews])),varI_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		        tau0_varT = [ktau(range(len(varT_all[i][:start_idx_ews])),varT_all[i][:start_idx_ews])[0] for i in range(samples)]
		        tau_varT = [ktau(range(len(varT_all[i][start_idx_ews:cut_idx_ews])),varT_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		        tau0_dcca = [ktau(range(len(dcca_all[i][:start_idx_ews])),dcca_all[i][:start_idx_ews])[0] for i in range(samples)]
		        tau_dcca = [ktau(range(len(dcca_all[i][start_idx_ews:cut_idx_ews])),dcca_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		        tau0_cc = [ktau(range(len(cc_all[i][:start_idx_ews])),cc_all[i][:start_idx_ews])[0] for i in range(samples)]
		        tau_cc = [ktau(range(len(cc_all[i][start_idx_ews:cut_idx_ews])),cc_all[i][start_idx_ews:cut_idx_ews])[0] for i in range(samples)]

		        ac1I_tau_med[k][j] = np.median(tau_ac1I)
		        ac1T_tau_med[k][j] = np.median(tau_ac1T)
		        varI_tau_med[k][j] = np.median(tau_varI)
		        varT_tau_med[k][j] = np.median(tau_varT)
		        dcca_tau_med[k][j] = np.median(tau_dcca)
		        cc_tau_med[k][j] = np.median(tau_cc)

		        ac1I_tau_medS[k][j] = comp_sign(np.percentile(tau_ac1I,5.), np.percentile(tau0_ac1I,95.))
		        ac1T_tau_medS[k][j] = comp_sign(np.percentile(tau_ac1T,5.), np.percentile(tau0_ac1T,95.))
		        varI_tau_medS[k][j] = comp_sign(np.percentile(tau_varI,5.), np.percentile(tau0_varI,95.))
		        varT_tau_medS[k][j] = comp_sign(np.percentile(tau_varT,5.), np.percentile(tau0_varT,95.))
		        dcca_tau_medS[k][j] = comp_sign(np.percentile(tau_dcca,5.), np.percentile(tau0_dcca,95.))
		        cc_tau_medS[k][j] = comp_sign(np.percentile(tau_cc,5.), np.percentile(tau0_cc,95.))

		        print(k, j)
		        print(np.median(tau_ac1I), np.percentile(tau_ac1I,5.), np.percentile(tau0_ac1I,95.))
		        print(np.median(tau_ac1T), np.percentile(tau_ac1T,5.), np.percentile(tau0_ac1T,95.))
		        print(np.median(tau_varI), np.percentile(tau_varI,5.), np.percentile(tau0_varI,95.))
		        print(np.median(tau_varT), np.percentile(tau_varT,5.), np.percentile(tau0_varT,95.))
		        print(np.median(tau_dcca), np.percentile(tau_dcca,5.), np.percentile(tau0_dcca,95.))
		        print(np.median(tau_cc), np.percentile(tau_cc,5.), np.percentile(tau0_cc,95.))
		        print('-------------------------------')


	to_file = np.asarray([ac1I_tau_med,ac1T_tau_med,varI_tau_med,varT_tau_med,dcca_tau_med,cc_tau_med])
	np.save('median_data2',to_file)
	to_file = np.asarray([ac1I_tau_medS,ac1T_tau_medS,varI_tau_medS,varT_tau_medS,dcca_tau_medS,cc_tau_medS])
	np.save('significance_data2',to_file)

	plot_significance(ac1I_tau_med,ac1I_tau_medS,wind_vals,filt)
	plot_significance(ac1T_tau_med,ac1T_tau_medS,wind_vals,filt)
	plot_significance(varI_tau_med,varI_tau_medS,wind_vals,filt)
	plot_significance(varT_tau_med,varT_tau_medS,wind_vals,filt)
	plot_significance(dcca_tau_med,dcca_tau_medS,wind_vals,filt)
	plot_significance(cc_tau_med,cc_tau_medS,wind_vals,filt)


def comp_sign(x,y):
        if x>y:
                return 1.
        else:
                return 0.

def stack_grids():

        wind_vals = np.linspace(50,380,16)
        filt = np.linspace(30,105,16)

        ac1I_tau_med = np.empty((len(wind_vals),len(filt))); ac1T_tau_med = np.empty((len(wind_vals),len(filt)))
        varI_tau_med = np.empty((len(wind_vals),len(filt))); varT_tau_med = np.empty((len(wind_vals),len(filt)))
        dcca_tau_med = np.empty((len(wind_vals),len(filt))); cc_tau_med = np.empty((len(wind_vals),len(filt)))

        ac1I_tau_medS = np.empty((len(wind_vals),len(filt))); ac1T_tau_medS = np.empty((len(wind_vals),len(filt)))
        varI_tau_medS = np.empty((len(wind_vals),len(filt))); varT_tau_medS = np.empty((len(wind_vals),len(filt)))
        dcca_tau_medS = np.empty((len(wind_vals),len(filt))); cc_tau_medS = np.empty((len(wind_vals),len(filt)))

        ac1I_tau_med0,ac1T_tau_med0,varI_tau_med0,varT_tau_med0,dcca_tau_med0,cc_tau_med0 = np.load('median_data.npy')
        ac1I_tau_medS0,ac1T_tau_medS0,varI_tau_medS0,varT_tau_medS0,dcca_tau_medS0,cc_tau_medS0 = np.load('significance_data.npy')

        for i in range(8):
                for j in range(8):
                        ac1I_tau_med[2*i][2*j] = ac1I_tau_med0[i][j]
                        ac1T_tau_med[2*i][2*j] = ac1T_tau_med0[i][j]
                        varI_tau_med[2*i][2*j] = varI_tau_med0[i][j]
                        varT_tau_med[2*i][2*j] = varT_tau_med0[i][j]
                        dcca_tau_med[2*i][2*j] = dcca_tau_med0[i][j]
                        cc_tau_med[2*i][2*j] = cc_tau_med[i][j]

                        ac1I_tau_medS[2*i][2*j] = ac1I_tau_medS0[i][j]
                        ac1T_tau_medS[2*i][2*j] = ac1T_tau_medS0[i][j]
                        varI_tau_medS[2*i][2*j] = varI_tau_medS0[i][j]
                        varT_tau_medS[2*i][2*j] = varT_tau_medS0[i][j]
                        dcca_tau_medS[2*i][2*j] = dcca_tau_medS0[i][j]
                        cc_tau_medS[2*i][2*j] = cc_tau_medS0[i][j]

        ac1I_tau_med0,ac1T_tau_med0,varI_tau_med0,varT_tau_med0,dcca_tau_med0,cc_tau_med0 = np.load('median_data12.npy')
        ac1I_tau_medS0,ac1T_tau_medS0,varI_tau_medS0,varT_tau_medS0,dcca_tau_medS0,cc_tau_medS0 = np.load('significance_data12.npy')

        for i in range(8):
                for j in range(8):
                        ac1I_tau_med[2*i][2*j+1] = ac1I_tau_med0[i][j]
                        ac1T_tau_med[2*i][2*j+1] = ac1T_tau_med0[i][j]
                        varI_tau_med[2*i][2*j+1] = varI_tau_med0[i][j]
                        varT_tau_med[2*i][2*j+1] = varT_tau_med0[i][j]
                        dcca_tau_med[2*i][2*j+1] = dcca_tau_med0[i][j]
                        cc_tau_med[2*i][2*j+1] = cc_tau_med[i][j]

                        ac1I_tau_medS[2*i][2*j+1] = ac1I_tau_medS0[i][j]
                        ac1T_tau_medS[2*i][2*j+1] = ac1T_tau_medS0[i][j]
                        varI_tau_medS[2*i][2*j+1] = varI_tau_medS0[i][j]
                        varT_tau_medS[2*i][2*j+1] = varT_tau_medS0[i][j]
                        dcca_tau_medS[2*i][2*j+1] = dcca_tau_medS0[i][j]
                        cc_tau_medS[2*i][2*j+1] = cc_tau_medS0[i][j]

        ac1I_tau_med0,ac1T_tau_med0,varI_tau_med0,varT_tau_med0,dcca_tau_med0,cc_tau_med0 = np.load('median_data21.npy')
        ac1I_tau_medS0,ac1T_tau_medS0,varI_tau_medS0,varT_tau_medS0,dcca_tau_medS0,cc_tau_medS0 = np.load('significance_data21.npy')

        for i in range(8):
                for j in range(8):
                        ac1I_tau_med[2*i+1][2*j] = ac1I_tau_med0[i][j]
                        ac1T_tau_med[2*i+1][2*j] = ac1T_tau_med0[i][j]
                        varI_tau_med[2*i+1][2*j] = varI_tau_med0[i][j]
                        varT_tau_med[2*i+1][2*j] = varT_tau_med0[i][j]
                        dcca_tau_med[2*i+1][2*j] = dcca_tau_med0[i][j]
                        cc_tau_med[2*i+1][2*j] = cc_tau_med[i][j]

                        ac1I_tau_medS[2*i+1][2*j] = ac1I_tau_medS0[i][j]
                        ac1T_tau_medS[2*i+1][2*j] = ac1T_tau_medS0[i][j]
                        varI_tau_medS[2*i+1][2*j] = varI_tau_medS0[i][j]
                        varT_tau_medS[2*i+1][2*j] = varT_tau_medS0[i][j]
                        dcca_tau_medS[2*i+1][2*j] = dcca_tau_medS0[i][j]
                        cc_tau_medS[2*i+1][2*j] = cc_tau_medS0[i][j]

        ac1I_tau_med0,ac1T_tau_med0,varI_tau_med0,varT_tau_med0,dcca_tau_med0,cc_tau_med0 = np.load('median_data22.npy')
        ac1I_tau_medS0,ac1T_tau_medS0,varI_tau_medS0,varT_tau_medS0,dcca_tau_medS0,cc_tau_medS0 = np.load('significance_data22.npy')

        for i in range(8):
                for j in range(8):
                        ac1I_tau_med[2*i+1][2*j+1] = ac1I_tau_med0[i][j]
                        ac1T_tau_med[2*i+1][2*j+1] = ac1T_tau_med0[i][j]
                        varI_tau_med[2*i+1][2*j+1] = varI_tau_med0[i][j]
                        varT_tau_med[2*i+1][2*j+1] = varT_tau_med0[i][j]
                        dcca_tau_med[2*i+1][2*j+1] = dcca_tau_med0[i][j]
                        cc_tau_med[2*i+1][2*j+1] = cc_tau_med[i][j]

                        ac1I_tau_medS[2*i+1][2*j+1] = ac1I_tau_medS0[i][j]
                        ac1T_tau_medS[2*i+1][2*j+1] = ac1T_tau_medS0[i][j]
                        varI_tau_medS[2*i+1][2*j+1] = varI_tau_medS0[i][j]
                        varT_tau_medS[2*i+1][2*j+1] = varT_tau_medS0[i][j]
                        dcca_tau_medS[2*i+1][2*j+1] = dcca_tau_medS0[i][j]
                        cc_tau_medS[2*i+1][2*j+1] = cc_tau_medS0[i][j]

        plot_significance(ac1I_tau_med,ac1I_tau_medS,wind_vals,filt)
        plot_significance(ac1T_tau_med,ac1T_tau_medS,wind_vals,filt)
        plot_significance(varI_tau_med,varI_tau_medS,wind_vals,filt)
        plot_significance(varT_tau_med,varT_tau_medS,wind_vals,filt)
        plot_significance(dcca_tau_med,dcca_tau_medS,wind_vals,filt)
        plot_significance(cc_tau_med,cc_tau_medS,wind_vals,filt)


def plot_significance(x,sig,wind,filt):

        X,Y = np.mgrid[0:len(wind)+1,0:len(filt)+1]
        fig1=pl.figure(figsize=(8,6))
        ax=pl.subplot(111)
        ax.tick_params(direction='out', length=6, width=1.5)

        pl.imshow(x, interpolation='nearest', cmap='RdBu', origin='lower',norm=MidpointNormalize(midpoint=0.0,vmin=np.min(x),vmax=np.max(x)) , extent=(min(filt),max(filt),min(wind),max(wind)),aspect="auto")
        pl.xlabel('Filter width'); pl.ylabel('Window size')
        pl.colorbar(label='Median $\\tau$')

        X,Y = np.mgrid[0:len(wind)+1,0:len(filt)+1]
        fig1=pl.figure(figsize=(8,6))
        ax=pl.subplot(111)
        pl.imshow(sig, interpolation='nearest', cmap='RdBu', origin='lower',norm=MidpointNormalize(midpoint=0.0,vmin=np.min(sig),vmax=np.max(sig)) , extent=(min(filt),max(filt),min(wind),max(wind)),aspect="auto")
        pl.colorbar()

def early_warning(wind=150, downs=1, sigI=0.02, count=100):

	theory_curv = np.load('eis12_theory_curv_h05.npy')

	h = 0.0005; loop = 150.
	T = 3100 #simulation time in years
	N_T = int(round(float(T)/h/loop)); t = np.linspace(0, T, N_T)
	#output sample spacing: h*loop (years)

	delt = 0.43 #albedo difference ocean - sea ice
	h_alph = 0.5
	F = 1/28. #ocean forcing on sea ice
	R0 = -0.1 #sea ice export in %
	B = 0.45 #outgoing longwave radiation coeff
	Lm = 1.25 #incoming longwave = greenhouse
	R=0.0 #sea ice import bifurcation parameter

	eta1 = 3.0; eta2 = 1.; eta3 = 0.3
	Icov = 1.156 #fixed point for h=0.5: 1.156
	fact = 0.35 #determines size of jump in eta1
	B0 = 1./Icov*fact
	RT = 1/200. #inverse timescale of ocean model
	sigT = 0.2
	sigS = 0.2

	start = 300
	dur = 350
	ampl = 0.3
	ampl2 = 0.15
	wait = 200
	### Timescale for theoretical indicators
	t_fin = 0.54/ampl*dur + start - wind # time when theor. curv stops
	t_fin_notip = 0.54/ampl*500 + start - wind # time when theor. curv stops
	bif_point = -0.281682336467 #bifurcation point for R with h=0.5
	#bif_point = -0.542268453691 #bifurcation point for R with h=0.08
	t_bif = -bif_point/ampl*dur + start  #time when bifurcation is passed
	t_bif_idx = np.argmin(np.abs(t - t_bif))

	t_bif_notip = -bif_point/ampl*500 + start  #time when bifurcation is passed
	t_bif_idx_notip = np.argmin(np.abs(t - t_bif_notip))

	theory_ac1 = np.exp(-theory_curv*h*loop*downs*50)
	theory_var = sigI**2/(2*theory_curv)

	ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,-ampl,int(dur/h)),int((T-start-dur)/h+loop)*[-ampl]),)
	ramp_downs = np.concatenate((int(start/h/loop)*[0.],np.linspace(0.,-ampl,int(dur/h/loop)),int((T-start-dur)/h/loop)*[-ampl]),)

	params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT,sigI,sigT,sigS])

	U_0 = [1.156,2.4,2.5]
	I,Te,S = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
	'''
	###Calculate correlation time
	cut = 1000; cut_t = int(round(float(start)/h/loop))
	autoco_I = ac(I[:cut_t], cut)
	ac_critval = 1.96/np.sqrt(cut_t)
	idx = np.argmax(autoco_I<ac_critval)
	print('correlation time (y): ', idx*h*loop)
	fig=pl.figure()
	pl.plot(np.linspace(0,start,cut),autoco_I)
	pl.plot(np.linspace(0,start,cut),cut*[ac_critval],'--',color='gray')
	pl.plot(np.linspace(0,start,cut),cut*[-ac_critval],'--',color='gray')
	pl.xlabel('lag (y)');pl.ylabel('autocorrelation')
	'''

	thin = 50#20#10
	filt = 0.0125/100#0.01/100 # Filter cut-off: fraction of 0.5 = 1/timestep

	window = int(wind/(h*loop)) #running window length in output sample timesteps
	print(window)

	samples = 300
	cut = 15000;t_kernel_g_I = 50; t_kernel_g_T = 50
	ac1_all=[];ac1T_all=[];cc_all=[];cc_tip=[];cc_notip=[];var_all=[];varT_all=[];varT_tip=[];rtip_count=0;I_align_all=[];T_align_all=[];S_align_all=[];var_all3=[];var_all7=[];ac1_all3=[];ac1_all7=[];dfa_all=[];dfa_all3=[];dfa_all7=[];ac1_allG=[];var_allG=[];dfa_all=[];dcca_all=[];tip_times=[]; adf_all=[]
	for i in range(samples):
		ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,-ampl,int(dur/h)),int((T-start-dur)/h+loop)*[-ampl]),)
		start_time = time.time()
		I0,Te0,S0 = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
		print('simulation time (s)', time.time()-start_time)
		idx=len(I0)
		idx = np.argmax(I0<0.0)
		tip_times.append(idx*h*loop)


		if (Te0[-1]>S0[-1] and Te0[-2]>S0[-2] and Te0[-3]>S0[-3] and Te0[-4]>S0[-4]):
			rtip_count += 1
			I_align = I0
			I_align_all.append(I_align)
			T_align = Te0
			T_align_all.append(T_align)
			S_align = S0
			S_align_all.append(S_align)
			start_time = time.time()
			acI,varI,acT,varT,dcca = calc_ews_spline(I_align,T_align,int(window),thin,8,3)
			ac1_allG.append(acI); var_allG.append(varI)
			ac1T_all.append(acT); varT_all.append(varT)
			dcca_all.append(dcca)
			print('EWS calc time (s)', time.time()-start_time)
		
	#np.save('seaice_ocean_ews_cubic_data.npy',np.asarray([var_allG, ac1_allG, varT_all, ac1T_all, dcca_all]))
	#np.save('seaice_ocean_cc_cubic_data_sigT02.npy',np.asarray([dcca_all]))
	#np.save('seaice_ocean_cubic_data_sigT02.npy',np.asarray([varT_all, ac1T_all, dcca_all]))
	
	print('fraction of R-tippings :' , rtip_count/float(samples))
	print('average, sdev, min, max time of I tipping: ', np.mean(tip_times),np.std(tip_times),min(tip_times),max(tip_times))
	
	
	ews_cut_time1 = 630
	ews_cut_time2 = 630+wind


	cut_idx = len(I_align)
	cut_idx_ews = int((cut_idx-window)/thin)
	mean,percentile_5,percentile_95 = calc_mean_conf(I_align_all,cut_idx,rtip_count)#cut
	meanT,percentile_5T,percentile_95T = calc_mean_conf(T_align_all,cut_idx,rtip_count)#cut
	meanS,percentile_5S,percentile_95S = calc_mean_conf(S_align_all,cut_idx,rtip_count)#cut
	
	### ensemble variance:
	var_T_ens = [np.std([T_align_all[i][j] for i in range(len(T_align_all))])**2 for j in range(len(T_align_all[0]))]



	mean_ac1G,percentile_5_ac1G,percentile_95_ac1G = calc_mean_conf(ac1_allG,cut_idx_ews,rtip_count)
	mean_ac1T,percentile_5_ac1T,percentile_95_ac1T = calc_mean_conf(ac1T_all,cut_idx_ews,rtip_count)
	mean_varG,percentile_5_varG,percentile_95_varG = calc_mean_conf(var_allG,cut_idx_ews,rtip_count)
	mean_varT,percentile_5_varT,percentile_95_varT = calc_mean_conf(varT_all,cut_idx_ews,rtip_count)
	#mean_dfa,percentile_5_dfa,percentile_95_dfa = calc_mean_conf(dfa_all,cut_idx_ews,samples)
	mean_dcca,percentile_5_dcca,percentile_95_dcca = calc_mean_conf(dcca_all,cut_idx_ews,rtip_count)
	#mean_adf,percentile_5_adf,percentile_95_adf = calc_mean_conf(adf_all,cut_idx_ews,samples)

	#mean_cc_tip,percentile_2_cc_tip,percentile_97_cc_tip = calc_mean_conf(cc_tip,cut_idx_ews,rtip_count)
	#mean_cc_notip,percentile_2_cc_notip,percentile_97_cc_notip = calc_mean_conf(cc_notip,cut_idx_ews,samples-rtip_count)

	#mean_2_cc_tip = np.mean(percentile_2_cc_tip[:int((start-wind)/h/loop/thin)])
	#mean_97_cc_tip = np.mean(percentile_97_cc_tip[:int((start-wind)/h/loop/thin)])

	#mean_2_cc_notip = np.mean(percentile_2_cc_notip[:int((start-wind)/h/loop/thin)])
	#mean_97_cc_notip = np.mean(percentile_97_cc_notip[:int((start-wind)/h/loop/thin)])

	### mean percentiles before parameter shift.
	mean_5_var = np.mean(percentile_5_varG[:int((start-wind)/h/loop/thin)])
	mean_95_var = np.mean(percentile_95_varG[:int((start-wind)/h/loop/thin)])
	mean_5_ac1 = np.mean(percentile_5_ac1G[:int((start-wind)/h/loop/thin)])
	mean_95_ac1 = np.mean(percentile_95_ac1G[:int((start-wind)/h/loop/thin)])
	#mean_5_dfa = np.mean(percentile_5_dfa[:int((start-wind)/h/loop/thin)])
	#mean_95_dfa = np.mean(percentile_95_dfa[:int((start-wind)/h/loop/thin)])
	mean_5_dcca = np.mean(percentile_5_dcca[:int((start-wind)/h/loop/thin)])
	mean_95_dcca = np.mean(percentile_95_dcca[:int((start-wind)/h/loop/thin)])

	cut=cut_idx
	t_align = np.linspace(0,cut*h*loop,cut)
	t_ews = np.linspace(window*h*loop,cut*h*loop,cut_idx_ews)
	#Cut EWS so that they don't include filter artifacts at beginning of tipping.
	ec = 1#int(t_kernel_g_I/h/loop/thin)
	t_ews = t_ews[:-ec]

	fig=pl.figure(figsize=(7.,11.))
	pl.subplots_adjust(left=0.2, bottom=0.08, right=0.92, top=0.97, wspace=0.25, hspace=0.2)
	pl.suptitle('Window=%s y, sig.=%s, Ramp=%s y, Kern. st.dev.=%s y'%(int(window*h*loop),sigI,dur,t_kernel_g_I), fontsize=12)
	ax1 = pl.subplot(711)
	ax1.fill_between(t_align,percentile_5,percentile_95,alpha=0.2, color='black')
	ax1.plot(t_align, mean, color='black')
	ax1.set_ylabel('I')
	ax2 = ax1.twinx()
	ax2.plot(np.linspace(0,len(ramp_downs)*h*loop,len(ramp_downs)),ramp_downs,':',color=colors[4])
	ax2.set_ylabel('R')
	pl.xlim(0,cut*h*loop)

	pl.subplot(712)
	pl.fill_between(t_align,percentile_5S,percentile_95S,color='tomato',alpha=0.2)
	pl.fill_between(t_align,percentile_5T,percentile_95T, color='black',alpha=0.2)
	pl.plot(t_align, meanS,label='S', color='tomato')
	pl.plot(t_align, meanT,label='T', color='black')
	pl.xlim(0,cut*h*loop);pl.ylabel('T, S');pl.legend(loc='best', fontsize=12)

	pl.subplot(713)
	pl.fill_between(t_ews,percentile_5_varG[:-ec],percentile_95_varG[:-ec],alpha=0.2,label='90% conf.', color='black')
	pl.plot(t_ews,len(t_ews)*[mean_5_var],'--',color='gray')
	pl.plot(t_ews,len(t_ews)*[mean_95_var],'--',color='gray')
	pl.plot(t_ews, mean_varG[:-ec], color='black')
	pl.plot(100*[ews_cut_time1],np.linspace(min(mean_varG),max(mean_varG),100),':',color='tomato')
	pl.plot(100*[ews_cut_time2],np.linspace(min(mean_varG),max(mean_varG),100),':',color='tomato')
	pl.xlim(0,cut*h*loop);pl.legend(loc='upper left', fontsize=12);pl.ylabel('variance (I)')
	
	pl.subplot(714)
	pl.fill_between(t_ews,percentile_5_ac1G[:-ec],percentile_95_ac1G[:-ec],alpha=0.2, color='black')
	pl.plot(t_ews,len(t_ews)*[mean_5_ac1],'--',color='gray')
	pl.plot(t_ews,len(t_ews)*[mean_95_ac1],'--',color='gray')
	pl.plot(t_ews, mean_ac1G[:-ec], color='black')
	pl.plot(100*[ews_cut_time1],np.linspace(min(mean_ac1G),max(mean_ac1G),100),':',color='tomato')
	pl.plot(100*[ews_cut_time2],np.linspace(min(mean_ac1G),max(mean_ac1G),100),':',color='tomato')
	pl.xlim(0,cut*h*loop);pl.legend(loc='best');pl.ylabel('AR(1) coeff. (I)');pl.xlabel('time (y)')
	pl.subplot(715)
	pl.fill_between(t_ews,percentile_5_varT[:-ec],percentile_95_varT[:-ec],alpha=0.2, color='black')
	pl.plot(t_ews, mean_varT[:-ec], color='black')
	pl.plot(100*[ews_cut_time1],np.linspace(min(mean_varT),max(mean_varT),100),':',color='tomato')
	pl.plot(100*[ews_cut_time2],np.linspace(min(mean_varT),max(mean_varT),100),':',color='tomato')
	pl.xlim(0,cut*h*loop);pl.ylabel('variance (T)')
	
	pl.subplot(716)
	pl.fill_between(t_ews,percentile_5_ac1T[:-ec],percentile_95_ac1T[:-ec],alpha=0.2,color='black')
	pl.plot(t_ews, mean_ac1T[:-ec],color='black')
	pl.plot(100*[ews_cut_time1],np.linspace(min(mean_ac1T),max(mean_ac1T),100),':',color='tomato')
	pl.plot(100*[ews_cut_time2],np.linspace(min(mean_ac1T),max(mean_ac1T),100),':',color='tomato')
	pl.xlim(0,cut*h*loop);pl.ylabel('AR(1) coeff. (T)');pl.xlabel('time (y)')

	#pl.subplot(716)
	#pl.fill_between(t_ews,percentile_5_dfa[:-ec],percentile_95_dfa[:-ec],alpha=0.2, color='black')
	#pl.plot(t_ews,len(t_ews)*[mean_5_dfa],'--',color='gray')
	#pl.plot(t_ews,len(t_ews)*[mean_95_dfa],'--',color='gray')
	#pl.plot(t_ews, mean_dfa[:-ec], color='black')
	#pl.xlim(0,cut*h*loop);pl.legend(loc='best');pl.xlabel('time (y)')
	#pl.ylabel('DFA exp. (I)')
	#pl.ylabel('ADF (T)')

	#pl.subplot(717)
	#pl.fill_between(t_ews,percentile_5_dcca[:-ec],percentile_95_dcca[:-ec],alpha=0.2, color='black')
	#pl.plot(t_ews,len(t_ews)*[mean_5_dcca],'--',color='gray')
	#pl.plot(t_ews,len(t_ews)*[mean_95_dcca],'--',color='gray')
	#pl.plot(t_ews, mean_dcca[:-ec], color='black')
	#pl.plot(100*[ews_cut_time1],np.linspace(min(mean_dcca),max(mean_dcca),100),':',color='tomato')
	#pl.plot(100*[ews_cut_time2],np.linspace(min(mean_dcca),max(mean_dcca),100),':',color='tomato')
	#pl.xlim(0,cut*h*loop);pl.ylabel('DCCA exp.');pl.xlabel('time (y)')

	#pl.subplot(717)
	#pl.fill_between(t_ews,percentile_5_adf[:-ec],percentile_95_adf[:-ec],alpha=0.2, color='black')
	#pl.plot(t_ews,len(t_ews)*[mean_5_adf],'--',color='gray')
	#pl.plot(t_ews,len(t_ews)*[mean_95_adf],'--',color='gray')
	#pl.plot(t_ews, mean_adf[:-ec], color='black')
	#pl.xlim(0,cut*h*loop);pl.legend(loc='best');pl.ylabel('ADF detrend (T)');pl.xlabel('time (y)')
	
	pl.subplot(717)
	pl.plot(t_align,var_T_ens, color='tomato')
	
	'''
	cut_idx_notip = cut_idx_ews
	cut_idx = cut_idx_ews

	### Unit root test
	figE=pl.figure()
	pl.subplot(211)
	#pl.suptitle('Early-warning of AMOC R-tipping (Cross-correlation). Window=%s, sig.=%s, Filter Bandw.=%s'%(window*h*loop,sigT,0.02/filt))
	pl.suptitle('Early-warning of AMOC R-tipping (Unit root). Window=%s, sig.=%s, Filter Bandw.=%s'%(window*h*loop,sigT,0.02/filt))
	pl.fill_between(np.linspace(0,cut_idx_notip*downs*loop*h*thin,cut_idx_notip),percentile_2_cc_notip,percentile_97_cc_notip,alpha=0.2,label='95% confidence')
	pl.plot(np.linspace(0,cut_idx*loop*h*thin,cut_idx),mean_cc_tip,'--',label='ADF, Tip')
	pl.plot(np.linspace(0,cut_idx_notip*downs*loop*h*thin,cut_idx_notip),mean_cc_notip,label='ADF, No tip')
	#pl.plot(100*[t[t_bif_idx_notip]-wind],np.linspace(min(percentile_2_cc_notip),max(percentile_97_cc_notip),100),'--',color='gray',alpha=0.5)
	pl.plot(100*[ews_cut_time1],np.linspace(min(percentile_2_cc_notip),max(percentile_97_cc_notip),100),':',color='black')
	pl.plot(100*[ews_cut_time2],np.linspace(min(percentile_2_cc_notip),max(percentile_97_cc_notip),100),':',color='black')
	pl.plot(np.linspace(0, t_fin_notip, 100),100*[mean_2_cc_notip],color='gray',alpha=0.5)
	pl.plot(np.linspace(0, t_fin_notip, 100),100*[mean_97_cc_notip],color='gray',alpha=0.5)
	pl.legend(loc='best')

	pl.subplot(212)
	pl.fill_between(np.linspace(0,cut_idx*loop*h*thin,cut_idx),percentile_2_cc_tip,percentile_97_cc_tip,alpha=0.2,label='95% confidence')
	pl.plot(np.linspace(0,cut_idx*loop*h*thin,cut_idx),mean_cc_tip,label='ADF, Tip')
	#pl.plot(100*[t[t_bif_idx]-wind],np.linspace(min(percentile_2_cc_tip),max(percentile_97_cc_tip),100),'--',color='gray',alpha=0.5)
	pl.plot(100*[ews_cut_time1],np.linspace(min(percentile_2_cc_tip),max(percentile_97_cc_tip),100),':',color='black')
	pl.plot(100*[ews_cut_time2],np.linspace(min(percentile_2_cc_tip),max(percentile_97_cc_tip),100),':',color='black')
	pl.plot(np.linspace(0, t_fin, 100),100*[mean_2_cc_tip],color='gray',alpha=0.5)
	pl.plot(np.linspace(0, t_fin, 100),100*[mean_97_cc_tip],color='gray',alpha=0.5)
	pl.legend(loc='best')


	figD=pl.figure()
	pl.suptitle('Early-warning of AMOC R-tipping (variance). Window=%s, sig.=%s, Filter Bandw.=%s'%(2*window*h*loop,sigT,0.02/filt))
	pl.subplot(211)
	pl.fill_between(np.linspace(2*wind,T,len(varT)),percentile_2_var_T,percentile_97_var_T,alpha=0.2,label='95% confidence')
	pl.plot(np.linspace(2*wind,T,len(varT)),mean_var_T,label='var(T), No tip')
	pl.plot(100*[t[t_bif_idx_notip]],np.linspace(min(percentile_2_var_T),max(percentile_97_var_T),100),'--',color='gray',alpha=0.5)
	pl.plot(np.linspace(2*wind,T, 100),100*[mean_2_var_T],color='gray',alpha=0.5)#t_fin+wind
	pl.plot(np.linspace(2*wind,T, 100),100*[mean_97_var_T],color='gray',alpha=0.5)
	pl.legend(loc='best')
	pl.ylabel('var(T)')
	pl.xlabel('time (years)')

	pl.subplot(212)
	pl.fill_between(np.linspace(2*wind,T,len(varT)),percentile_2_var_T_tip,percentile_97_var_T_tip,alpha=0.2,label='95% confidence')
	pl.plot(np.linspace(2*wind,T,len(varT)),mean_var_T_tip,label='var(T), Tip')
	pl.plot(100*[t[t_bif_idx]],np.linspace(min(percentile_2_var_T_tip),max(percentile_97_var_T_tip),100),'--',color='gray',alpha=0.5)
	pl.plot(np.linspace(2*wind, T, 100),100*[mean_2_var_T_tip],color='gray',alpha=0.5)
	pl.plot(np.linspace(2*wind, T, 100),100*[mean_97_var_T_tip],color='gray',alpha=0.5)
	pl.legend(loc='best')
	pl.ylabel('var(T)')
	pl.xlabel('time (years)')

	### Realizations aligned to start
	figY=pl.figure()
	pl.subplot(311)
	pl.suptitle('Early-warning of sea ice collapse. Window=%s, sig.=%s, Filter Bandw.=%s \n Realizations aligned to start'%(window*h*loop,sigI,0.02/filt))#downs
	pl.fill_between(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),percentile_2_ac1_align,percentile_97_ac1_align,alpha=0.2,label='95% confidence')
	pl.plot(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),mean_ac1_align,label='AR(1 year)')
	pl.plot(np.linspace(start-wind, t_fin, len(theory_ac1)),theory_ac1,color='crimson',label='theory')
	pl.plot(np.linspace(0,start-wind, 100),100*[theory_ac1[0]],color='crimson')
	pl.plot(100*[t[t_bif_idx]-wind],np.linspace(min(percentile_2_ac1_align),max(percentile_97_ac1_align),100),'--',color='gray',alpha=0.5)
	pl.plot(np.linspace(0, t_fin, 100),100*[mean_2_ac1_align],color='gray',alpha=0.5)
	pl.plot(np.linspace(0, t_fin, 100),100*[mean_97_ac1_align],color='gray',alpha=0.5)
	pl.legend(loc='best')

	pl.subplot(312)
	pl.fill_between(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),percentile_2_var_align,percentile_97_var_align,alpha=0.2,label='95% confidence')
	pl.plot(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),mean_var_align,label='var(I)')
	pl.plot(np.linspace(start-wind, t_fin, len(theory_var)),theory_var,color='crimson',label='theory')
	pl.plot(np.linspace(0,start-wind, 100),100*[theory_var[0]],color='crimson')
	pl.plot(100*[t[t_bif_idx]-wind],np.linspace(min(percentile_2_var_align),max(percentile_97_var_align),100),'--',color='gray',alpha=0.5)
	pl.plot(np.linspace(0, t_fin, 100),100*[mean_2_var_align],color='gray',alpha=0.5)
	pl.plot(np.linspace(0, t_fin, 100),100*[mean_97_var_align],color='gray',alpha=0.5)
	pl.legend(loc='best')

	pl.subplot(313)
	pl.fill_between(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),percentile_2_cc_align,percentile_97_cc_align,alpha=0.2,label='95% confidence')
	pl.plot(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),mean_cc_align,label='XC(I,T)')
	pl.plot(100*[t[t_bif_idx]-wind],np.linspace(min(percentile_2_cc_align),max(percentile_97_cc_align),100),'--',color='gray',alpha=0.5)
	pl.plot(np.linspace(0, t_fin, 100),100*[mean_2_cc_align],color='gray',alpha=0.5)
	pl.plot(np.linspace(0, t_fin, 100),100*[mean_97_cc_align],color='gray',alpha=0.5)
	pl.legend(loc='best')

	fig1=pl.figure()
	pl.subplot(511)
	pl.plot(100*[start],np.linspace(min(I),max(I),100),'--',color='gray',alpha=0.5)
	pl.plot(100*[start+dur],np.linspace(min(I),max(I),100),'--',color='gray',alpha=0.5)
	pl.plot(100*[t[t_bif_idx]],np.linspace(min(I),max(I),100),'--',color='gray',alpha=0.5)
	pl.plot(t,I,label='Ice')
	pl.plot(t[:idx1],I_detr0,label='Ice filtered')
	pl.xlim(-30,T+30)
	pl.legend(loc='best')
	pl.subplot(512)
	pl.plot(t,Te,label='T')
	pl.plot(t,Te_detr+2.45,label='T filtered')
	pl.plot(t,S,'--',label='S')
	pl.plot(t, eta1-B0*np.heaviside(I,0.5)*I,label='eta1')
	pl.legend(loc='best')
	pl.xlim(-30,T+30)
	pl.subplot(513)
	#pl.plot(t,[R_ramp(x) for x in t],label='Import')
	pl.plot(t,ramp[::40],label='Import')#ramp_ext[::40]
	pl.plot(100*[t[t_bif_idx]],np.linspace(min(ramp),max(ramp),100),'--',color='gray',alpha=0.5)
	pl.legend(loc='best')
	pl.xlim(-30,T+30)
	pl.subplot(514)
	pl.plot(t,Te-S,label='q')
	pl.plot(100*[start],np.linspace(min(Te-S),max(Te-S),100),'--',color='gray',alpha=0.5)
	pl.plot(100*[start+dur],np.linspace(min(Te-S),max(Te-S),100),'--',color='gray',alpha=0.5)
	pl.plot(100*[t[t_bif_idx]],np.linspace(min(Te-S),max(Te-S),100),'--',color='gray',alpha=0.5)
	#pl.plot(100*[start+dur+wait],np.linspace(min(Te-S),max(Te-S),100),'--',color='gray',alpha=0.5)
	#pl.plot(100*[start+2*dur+wait],np.linspace(min(Te-S),max(Te-S),100),'--',color='gray',alpha=0.5)
	pl.xlim(-30,T+30)
	pl.legend(loc='best')
	pl.subplot(515)
	t_ews = t[window:idx1:thin]
	pl.plot(t_ews[:-1],ac1_I,label='Running AC(1y) I')#[window::downs]#[:-1]
	pl.plot(t_ews[:-1],cc_IT,label='Running XC(I,T)')
	pl.plot(t_ews[:-1],1000*np.asarray(var_I),label='Running var(I)')

	pl.plot(t[window::thin],ac1_T,label='Running AC(1y) T')#[window::downs]#[:-1]
	#pl.plot(t[window::thin],cc_TI,label='Running XC(T,I)')
	pl.plot(t[window::thin],1000*np.asarray(var_T),label='Running var(T)')

	pl.plot(100*[t[t_bif_idx]],np.linspace(min(cc_IT),max(ac1_I),100),'--',color='gray',alpha=0.5)
	pl.legend(loc='best')
	pl.xlim(-30,T+30)
	pl.xlabel('time (years)')
	'''

### Return EWS indicators of data in running window
def calc_ews_filtered(data1,data2,window,thin,lag):
        n = thin
        return [[estimated_ac1_alt2(data1[n*i:n*i+window]) for i in range(int((len(data1)-window)/n))], [ews.cy_var(data1[n*i:n*i+window]) for i in range(int((len(data1)-window)/n))] , [estimated_ac1_alt2(data2[n*i:n*i+window]) for i in range(int((len(data1)-window)/n))], [ews.cy_var(data2[n*i:n*i+window]) for i in range(int((len(data1)-window)/n))], [DCCA(data1[n*i:n*i+window],data2[n*i:n*i+window]) for i in range(int((len(data1)-window)/n))]]
        
def calc_ews_spline(data1,data2,window,thin,lag, order):
        n = thin
        return [[estimated_ac1_alt2(detrend_spline(data1[n*i:n*i+window],order)) for i in range(int((len(data1)-window)/n))], [ews.cy_var(detrend_spline(data1[n*i:n*i+window],order)) for i in range(int((len(data1)-window)/n))] , [estimated_ac1_alt2(detrend_spline(data2[n*i:n*i+window],order)) for i in range(int((len(data1)-window)/n))], [ews.cy_var(detrend_spline(data2[n*i:n*i+window],order)) for i in range(int((len(data1)-window)/n))], [ews.estimated_cc(detrend_spline(data1[n*i:n*i+window], order),detrend_spline(data2[n*i:n*i+window], order)) for i in range(int((len(data1)-window)/n))]]

def calc_ews_sign(data1,data2,window,thin):
        n = thin
        return [[estimated_ac1_alt2(data1[n*i:n*i+window]) for i in range((len(data1)-window)/n)], [ews.cy_var(data1[n*i:n*i+window]) for i in range((len(data1)-window)/n)] , [estimated_ac1_alt2(data2[n*i:n*i+window]) for i in range((len(data2)-window)/n)], [ews.cy_var(data2[n*i:n*i+window]) for i in range((len(data2)-window)/n)] ,[DCCA(data1[n*i:n*i+window],data2[n*i:n*i+window]) for i in range((len(data1)-window)/n)],[ews.estimated_cc(data1[n*i:n*i+window],data2[n*i:n*i+window]) for i in range((len(data1)-window)/n)] ]


### Return EWS indicators of data detrended with polynomial spline in running window
def calc_ews2(data,window,n,lag,order):
        return [[ews.estimated_ac1(detrend_spline(data[n*i:n*i+window],order),lag) for i in range((len(data)-window)/n)],[ews.cy_var(detrend_spline(data[n*i:n*i+window],order)) for i in range((len(data)-window)/n)]]


def simulate():

	h = 0.0005
	loop = 500.
	T = 5000 #simulation time in years
	N_T = int(round(float(T)/h/loop))
	t = np.linspace(0, T, N_T)

	delt = 0.43 #albedo difference ocean - sea ice
	h_alph = 0.5
	F = 1/28. #ocean forcing on sea ice
	R0 = -0.1 #sea ice export in %
	B = 0.45 #outgoing longwave radiation coeff
	Lm = 1.25 #incoming longwave = greenhouse
	R=0. #sea ice import bifurcation parameter
	sigI = 0.

	eta1 = 3.0
	eta2 = 1.
	eta3 = 0.3
	Icov = 1.1715
	fact = 0.35
	B0 = 1./Icov*fact
	RT = 1/200. #inverse timescale of ocean model
	sigT = 0.
	sigS = 0.

	start = 1000
	dur = 340
	ampl = 0.3
	ampl2 = 0.15
	wait = 200

	ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,-ampl,int(dur/h)),int((T-start-dur)/h)*[-ampl]),)

	params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT,sigI,sigT,sigS])

	U_0 = [1.1,2.4,2.5]
	start_time = time.time()
	I,Te,S = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
	print('simulation time (s)', time.time()-start_time)
	to_file = t,ramp[::int(loop)],I,Te,S
	fig1=pl.figure()
	pl.subplot(411)
	pl.plot(t,I,label='Ice')
	pl.legend(loc='best')
	pl.subplot(412)
	pl.plot(t,Te,label='T')
	pl.plot(t,S,'--',label='S')
	pl.plot(t, eta1-B0*np.heaviside(I,0.5)*I,label='eta1')
	pl.legend(loc='best')
	pl.subplot(413)
	pl.plot(t,ramp[::int(loop)],label='Import R(t)')
	pl.legend(loc='best')
	pl.subplot(414)
	pl.plot(t,Te-S,label='q')
	pl.plot(100*[start],np.linspace(min(Te-S),max(Te-S),100),'--',color='gray',alpha=0.5)
	pl.plot(100*[start+dur],np.linspace(min(Te-S),max(Te-S),100),'--',color='gray',alpha=0.5)
	pl.legend(loc='best')
	pl.xlabel('time (years)')
	
	[R_on, I_on], [R_off, I_off], [R_ustab, I_ustab] = np.load('eis12_bif_data_h05.npy', allow_pickle=True)
		
	node_warm = np.loadtxt('stommel_bif_warm.txt',unpack=True)
	node_cold = np.loadtxt('stommel_bif_cold.txt',unpack=True)
	node_unstable = np.loadtxt('stommel_bif_unstable.txt',unpack=True)

	spl = UnivariateSpline(node_warm[0], node_warm[1], s=0, k=1)
	etagrid1 = np.arange(2.55, 3.7, 0.01)
	warm_intpT = spl(etagrid1)
	
	spl = UnivariateSpline(node_warm[0], node_warm[2], s=0, k=1)
	warm_intpS = spl(etagrid1)
	
	spl = UnivariateSpline(node_cold[0], node_cold[1], s=0, k=1)
	etagrid2 = np.arange(2.0, 3.34, 0.01)
	cold_intpT = spl(etagrid2)
	
	spl = UnivariateSpline(node_cold[0], node_cold[2], s=0, k=1)
	cold_intpS = spl(etagrid2)
	
	spl = UnivariateSpline(node_unstable[0][::-1], node_unstable[1][::-1], s=0, k=1)
	etagrid3 = np.arange(2.55, 3.34, 0.01)
	unstable_intpT = spl(etagrid3)
	
	spl = UnivariateSpline(node_unstable[0][::-1], node_unstable[2][::-1], s=0, k=1)
	unstable_intpS = spl(etagrid3)

	c1 = 5300
	c2 = 5400
	c3 = 6800

	R = ramp[::int(loop)]
	eta = eta1-B0*np.heaviside(I,0.5)*I
	
	fig=pl.figure(figsize=(9.,4.))
	pl.subplots_adjust(left=0.1, bottom=0.16, right=0.96, top=0.97, wspace=0.2, hspace=0.3)
	pl.subplot(121)
	pl.plot(R_on, I_on,color='tomato')
	pl.plot(R_off, I_off,color='black')
	pl.plot(R_ustab, I_ustab,':',color='black')
	pl.plot(R[:c1], I[:c1], color='forestgreen', linewidth=2.5)
	pl.plot(R[c1:], I[c1:], color='purple', linewidth=2.5)
	pl.xlabel('R')
	pl.ylabel('I')
	pl.subplot(122)
	pl.plot(etagrid1, warm_intpT,  color='black')
	pl.plot(etagrid2, cold_intpT, color='tomato')
	pl.plot(etagrid3, unstable_intpT, ':', color='black')
	pl.plot(eta[:c1], Te[:c1], color='forestgreen', linewidth=2.5)
	pl.plot(eta[c1:c2], Te[c1:c2], color='purple', linewidth=2.5)
	pl.plot(eta[c3:], Te[c3:], color='royalblue', linewidth=2.5)
	pl.plot(eta[c2:c3], Te[c2:c3], color='gold', linewidth=1.)
	pl.xlabel('$\eta_1 (I)$');pl.ylabel('T')
	pl.xlim(2.3,3.5)
	
	print(etagrid1)
	print(etagrid2)
	print(etagrid3)

	fig=pl.figure()
	pl.plot(cold_intpT[65:101], cold_intpS[65:101])
	pl.plot(warm_intpT[10:46], warm_intpS[10:46])
	
	pl.plot(cold_intpT[65], cold_intpS[65], 'o')
	pl.plot(warm_intpT[10], warm_intpS[10], 's')
	pl.plot(cold_intpT[100], cold_intpS[100], 'o')
	pl.plot(warm_intpT[45], warm_intpS[45], 's')
	
	pl.plot(unstable_intpT[10:46], unstable_intpS[10:46])
	
	pl.plot(unstable_intpT[10], unstable_intpS[10], '^')
	pl.plot(unstable_intpT[45], unstable_intpS[45], '^')
	
	pl.plot(Te, S, color='forestgreen', linewidth=2.5)

	pl.show()

	'''
	start_time = time.time()
	Warm_2 = []; Cold_2 = []; Warm_3 = []; Cold_3 = []        
	T0_values = np.linspace(1.,4.,22)#22
	S0_values = np.linspace(1.05,4.1,23)#23

	for i in range(len(T0_values)):
		for j in range(len(S0_values)):
		        eta1 =2.65
		        params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT])
		        U_0 = [0.2,T0_values[i],S0_values[j]] 
		        I,Te,S = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params)#= em_fast(f, U_0, h, T, 3, loop, 0.)
		        if Te[-1]>S[-1]:
		                Warm_2.append([T0_values[i],S0_values[j]])
		        else:
		                Cold_2.append([T0_values[i],S0_values[j]])

		        eta1 =3.
		        params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT])
		        I,Te,S = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params)
		        if Te[-1]>S[-1]:
		                Warm_3.append([T0_values[i],S0_values[j]])
		        else:
		                Cold_3.append([T0_values[i],S0_values[j]])

	print('simulation time (s)', time.time()-start_time)
	#print(len(I))

	### simulate to fixed points
	eta1 =2.65
	params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT])
	U_0 = [0.2,3.5,4.0]
	I,Te,S = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params)
	attr_2_1 = [Te[-1],S[-1]]
	U_0 = [0.2,3.,1.5]
	I2,Te2,S2 = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params)
	attr_2_2 = [Te2[-1],S2[-1]]

	eta1 =3.
	params = np.asarray([delt,h_alph,F,R0,B,Lm,R,eta1,eta2,eta3,B0,RT])
	U_0 = [0.2,3.9,4.0]
	I3,Te3,S3 = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params)
	attr_3_1 = [Te3[-1],S3[-1]]
	U_0 = [0.2,3.0,2.0]
	I4,Te4,S4 = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params)
	attr_3_2 = [Te4[-1],S4[-1]]
	U_0 = [0.2,2.75,3.75]
	I5,Te5,S5 = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params)
	attr_3_3 = [Te5[-1],S5[-1]]
	U_0 = [0.2,3.75,3.5]
	I6,Te6,S6 = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params)
	attr_3_4 = [Te6[-1],S6[-1]]

	fig2=pl.figure()
	pl.plot(S,Te)
	pl.xlabel('S')
	pl.ylabel('T')

	fig3=pl.figure()
	ax1=pl.subplot(121)
	ax1.set_title('eta1=2.65')
	pl.plot(np.linspace(1.,4.,100),np.linspace(1.,4.,100),'--',color='gray',alpha=0.6)
	for i in range(len(Warm_2)):
		pl.plot(Warm_2[i][0],Warm_2[i][1],'o',color='crimson',alpha=0.3)
	for i in range(len(Cold_2)):
		pl.plot(Cold_2[i][0],Cold_2[i][1],'o',color='midnightblue',alpha=0.3)
	pl.plot(attr_2_1[0],attr_2_1[1],'s',color='green')
	pl.plot(attr_2_2[0],attr_2_2[1],'s',color='green')
	pl.plot(Te,S)
	pl.plot(Te2,S2)
	pl.xlabel('T')
	pl.ylabel('S')

	ax2=pl.subplot(122)
	ax2.set_title('eta1=3')
	pl.plot(np.linspace(1.,4.,100),np.linspace(1.,4.,100),'--',color='gray',alpha=0.6)
	for i in range(len(Warm_3)):
		pl.plot(Warm_3[i][0],Warm_3[i][1],'o',color='crimson',alpha=0.3)
	for i in range(len(Cold_3)):
		pl.plot(Cold_3[i][0],Cold_3[i][1],'o',color='midnightblue',alpha=0.3)
	pl.plot(attr_3_1[0],attr_3_1[1],'s',color='green')
	pl.plot(attr_3_2[0],attr_3_2[1],'s',color='green')
	pl.plot(attr_3_3[0],attr_3_3[1],'s',color='green')
	pl.plot(attr_3_4[0],attr_3_4[1],'s',color='green')

	pl.plot(attr_2_1[0],attr_2_1[1],'p',color='orange',alpha=0.7)
	pl.plot(attr_2_2[0],attr_2_2[1],'p',color='orange',alpha=0.7)

	pl.plot(Te3,S3)
	pl.plot(Te4,S4)
	pl.plot(Te5,S5)
	pl.plot(Te6,S6)
	pl.xlabel('T')
	pl.ylabel('S')
	'''

def f(x,t):
        a = -1+delt*math.tanh(x[0]/h_alph) + (R0*np.heaviside(x[0],0.5)-B)*x[0] + R_ramp(t) - F + Lm
        b = eta1 - B0*np.heaviside(x[0],0.5)*x[0] - x[1] - np.abs(x[1]-x[2])*x[1]
        c = eta2 - eta3*x[2] - np.abs(x[1]-x[2])*x[2]
        return [a,RT*b,RT*c]

def R_ramp(t):
        if t<=start:
                return 0.
        elif start<t<start+dur:
                return -(t-start)*ampl/dur
        elif t>=start+dur:
                return -ampl

'''
def estimated_ac1(x):
        x = x-x.mean()
        lag = 25#1
        lenx = len(x)
        #r = np.array([np.sum(x[:lenx-1]*x[1:]) for k in range(n)])
        #r = np.sum(x[:lenx-1]*x[1:])
        #result = inv_variance*r/(np.arange(lenx, lenx-n, -1))
        result = 1/x.var()*np.sum(x[:lenx-lag]*x[lag:])/(lenx-lag)
        return result

def estimated_cc(x,y):
        x = x-x.mean()
        y = y-y.mean()
        lenx = len(x)
        result = 1/(np.sqrt(x.var())*np.sqrt(y.var()))*np.sum(x*y)/lenx
        return result

def estimated_ac1_alt(x):
        return np.corrcoef(x[:len(x)-1], x[1:])[0,1]
'''


        
'''
        ### trajectories aligned to where they hit given threshold. beginning aligned to start of shortest realization (first escape)
        mean_ac1 = [np.mean([ac1_all[j][len(ac1_all[j])-cut_idx+i] for j in range(samples)]) for i in range(cut_idx)]
        mean_cc = [np.mean([cc_all[j][len(ac1_all[j])-cut_idx+i] for j in range(samples)]) for i in range(cut_idx)]
        mean_var = [np.mean([var_all[j][len(ac1_all[j])-cut_idx+i] for j in range(samples)]) for i in range(cut_idx)]

        percentile_2_ac1 = [np.percentile([ac1_all[j][len(ac1_all[j])-cut_idx+i] for j in range(samples)],2.5) for i in range(cut_idx)]
        percentile_97_ac1 = [np.percentile([ac1_all[j][len(ac1_all[j])-cut_idx+i] for j in range(samples)],97.5) for i in range(cut_idx)]

        percentile_2_cc = [np.percentile([cc_all[j][len(cc_all[j])-cut_idx+i] for j in range(samples)],2.5) for i in range(cut_idx)]
        percentile_97_cc = [np.percentile([cc_all[j][len(cc_all[j])-cut_idx+i] for j in range(samples)],97.5) for i in range(cut_idx)]

        percentile_2_var = [np.percentile([var_all[j][len(var_all[j])-cut_idx+i] for j in range(samples)],2.5) for i in range(cut_idx)]
        percentile_97_var = [np.percentile([var_all[j][len(var_all[j])-cut_idx+i] for j in range(samples)],97.5) for i in range(cut_idx)]


        figX=pl.figure()
        pl.subplot(311)
        pl.suptitle('Early-warning of sea ice collapse. Window=%s, sigI.=%s, Filter Bandw.=%s\n Realizations aligned to end at threshold'%(window*h*loop,sigI,filt))#downs
        pl.fill_between(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),percentile_2_ac1,percentile_97_ac1,alpha=0.2,label='95% confidence')
        #for i in range(samples):
        #        pl.plot(np.linspace(0,cut_idx*downs*loop*h,cut_idx), ac1_all[i][len(ac1_all[i])-cut_idx:],color='gray',alpha=0.5,linewidth=0.5)
        pl.plot(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),mean_ac1,label='AR(1 year)')
        pl.plot(np.linspace(start-wind-offset, t_fin-offset, len(theory_ac1)),theory_ac1,color='crimson',label='theory')
        pl.plot(np.linspace(0,start-wind-offset, 100),100*[theory_ac1[0]],color='crimson')
        pl.legend(loc='best')

        pl.subplot(312)
        pl.fill_between(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),percentile_2_var,percentile_97_var,alpha=0.2,label='95% confidence')
        #for i in range(samples):
        #        pl.plot(np.linspace(0,cut_idx*downs*loop*h,cut_idx), var_all[i][len(ac1_all[i])-cut_idx:],color='gray',alpha=0.5,linewidth=0.5)
        pl.plot(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),mean_var,label='var(I)')
        pl.plot(np.linspace(start-wind-offset, t_fin-offset, len(theory_var)),theory_var,color='crimson',label='theory')
        pl.plot(np.linspace(0,start-wind-offset, 100),100*[theory_var[0]],color='crimson')
        pl.legend(loc='best')

        pl.subplot(313)
        pl.fill_between(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),percentile_2_cc,percentile_97_cc,alpha=0.2,label='95% confidence')
        #for i in range(samples):
        #        pl.plot(np.linspace(0,cut_idx*downs*loop*h,cut_idx), cc_all[i][len(ac1_all[i])-cut_idx:],color='gray',alpha=0.5,linewidth=0.5)
        pl.plot(np.linspace(0,cut_idx*downs*loop*h*thin,cut_idx),mean_cc,label='XC(I,T)')
        pl.legend(loc='best')
        #pl.savefig('earlywarning%s.pdf'%count)
'''

def calc_mean_conf(data,length,samples):
        mean = [np.mean([data[j][i] for j in range(samples)]) for i in range(length)]
        percentile_2 = [np.percentile([data[j][i] for j in range(samples)],5.) for i in range(length)]
        percentile_97 = [np.percentile([data[j][i] for j in range(samples)],95.) for i in range(length)]
        return mean, percentile_2, percentile_97

if __name__ == '__main__':
        MAIN()
