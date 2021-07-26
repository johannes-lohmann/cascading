import numpy as np
import matplotlib.pyplot as pl
from  matplotlib import colors
import stommel_cy as ode_cy
import stommel_cy_noise as sde_cy
import cmath
import ews_cy as ews
from scipy.stats import pearsonr
from scipy.ndimage import filters
from scipy.optimize import root
from scipy.signal import gaussian, detrend
from ews_functions import smoothing_filter_gauss, estimated_ac1_alt2
from scipy.interpolate import UnivariateSpline
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from ts_analysis import PermEn
import time
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from scipy.optimize import root
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib as mpl
import heapq

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

def MAIN():
        #T0,S0=simulate()
        #plot_real()
        #fixed_points()
        #critical_rate()
        linear_decay_rate()
        #earlywarning()
        #earlywarning_ensemble()
        #bif_diagram()
        #plot_realizations()
        #trans_prob()
        #tipping_times()
        #earlywarning_pullback_timeseries()
        pl.show()

def plot_real():
        t, T, S = np.loadtxt('stommel_rtip300y_real.txt')

        t1, T1, S1 = np.loadtxt('stommel_rtip500y_real.txt')

        st = 650
        pl.plot(T[st:],S[st:])
        pl.plot(T1[st:],S1[st:])

def bif_diagram():

        eta2 = 1.; eta3 = 0.3
        def fa(x):
                T = x[0]; S = x[1]
                return[-T+eta1 -np.abs(T-S)*T, -eta3*S + eta2 -np.abs(T-S)*S]

        eta1_values =np.linspace(2.6,3.3,2000)
        iguess1 = [2.35,2.5]
        iguess2 = [2.1,2.0]
        iguess3 = [0.5,0.4]
        branch1T=[];branch2T=[];branch3T=[];branch1S=[];branch2S=[];branch3S=[]; indi=0; bif_idx = len(eta1_values)
        for i in range(len(eta1_values)):
                eta1=eta1_values[i]
                sol1=root(fa,iguess1,jac=False)
                a1=sol1.x
                print(a1)
                sol2=root(fa,iguess2,jac=False)
                a2=sol2.x
                sol3=root(fa,iguess3,jac=False)
                a3=sol3.x
                branch1T.append(a1[0])
                branch2T.append(a2[0])
                branch3T.append(a3[0])
                branch1S.append(a1[1])
                branch2S.append(a2[1])
                branch3S.append(a3[1])
                iguess1=a1
                iguess2=a2
                iguess3=a3
                #if (a1[0]<2.0 and indi==0):
                #        bif_idx = i
                #        indi = 1
                #        print('bifurcation point: ', eta1_values[i])

        np.save('stommel_bif_fine',np.asarray([eta1_values,branch1T,branch1S,branch2T,branch2S,branch3T,branch3S]))

        fig2=pl.figure()
        pl.subplot(121)
        pl.plot(eta1_values,branch1T)
        pl.plot(eta1_values[:bif_idx],branch3T[:bif_idx])
        pl.plot(eta1_values[:bif_idx],branch2T[:bif_idx],':')
        pl.subplot(122)
        pl.plot(eta1_values,branch1S)
        pl.plot(eta1_values[:bif_idx],branch3S[:bif_idx])
        pl.plot(eta1_values[:bif_idx],branch2S[:bif_idx],':')


def earlywarning_ensemble():

        h = 0.05; loop = 40.0; T = 16000
        N_T = int(round(float(T)/h/loop))
        t = np.linspace(0, T, N_T)

        eta1 = 2.65; eta2 = 1.; eta3 = 0.3; RT = 1/200.; sigT=0.001; sigS=0.001
        params = np.asarray([eta1,eta2,eta3,RT,sigT,sigS])

        start = 10000; dur = 370; ampl = 0.35

        ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur/h)),int((T-start-dur)/h)*[ampl]),)

        U_0 = [3.,3.55]
        thin = 5
        samples=5

        Te_all = []; S_all = []; tip=0
        for i in range(samples):
                Te0,S0 = sde_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)

                if Te0[-1]>S0[-1]: #>
                        tip+=1
                        Te_all.append(Te0[::thin])
                        S_all.append(S0[::thin])

        print('fraction of R-tippings: ',float(tip)/samples)

        mean = [np.mean([Te_all[j][i] for j in range(tip)]) for i in range(len(Te_all[0]))]
        meanS = [np.mean([S_all[j][i] for j in range(tip)]) for i in range(len(S_all[0]))]
        variance = [np.var(np.asarray(Te_all)[:,i]) for i in range(len(Te_all[0]))]
        varianceS = [np.var(np.asarray(S_all)[:,i]) for i in range(len(S_all[0]))]
        autoco = [np.sum((np.asarray(Te_all)[:,i]-mean[i])*(np.asarray(Te_all)[:,i-1]-mean[i-1]))/np.sqrt(variance[i]*variance[i-1])/tip for i in range(1,len(Te_all[0]))]

        crossco = [np.sum((np.asarray(Te_all)[:,i]-mean[i])*(np.asarray(S_all)[:,i]-meanS[i]))/np.sqrt(variance[i]*varianceS[i])/tip for i in range(len(Te_all[0]))]

        percentile_2 = [np.percentile([Te_all[j][i] for j in range(tip)],2.5) for i in range(len(Te_all[0]))]
        percentile_97 = [np.percentile([Te_all[j][i] for j in range(tip)],97.5) for i in range(len(Te_all[0]))]

        percentile_2S = [np.percentile([S_all[j][i] for j in range(tip)],2.5) for i in range(len(S_all[0]))]
        percentile_97S = [np.percentile([S_all[j][i] for j in range(tip)],97.5) for i in range(len(S_all[0]))]

        t_thin = t[::thin]
        fig1=pl.figure()
        pl.suptitle('Stommel model: Ensemble EWS. R-tipping (%s). %s y ramp. Noise %s'%(float(tip)/samples,dur,sigT))
        pl.subplot(511)
        pl.fill_between(t_thin,percentile_2,percentile_97,alpha=0.2,label='95% confidence')
        pl.plot(t[::thin],mean)
        pl.plot(t,Te0)
        pl.ylabel('T')
        pl.subplot(512)
        pl.fill_between(t_thin,percentile_2S,percentile_97S,alpha=0.2,label='95% confidence')
        pl.plot(t[::thin],meanS)
        pl.plot(t,S0)
        pl.ylabel('S')
        pl.subplot(513)
        pl.plot(t_thin,variance)
        pl.ylabel('variance (T)')
        pl.subplot(514)
        pl.plot(t_thin[1:],autoco)
        pl.ylabel('AC (T)')
        pl.subplot(515)
        pl.plot(t_thin,crossco)
        pl.ylabel('XC (T,S)')
        pl.xlabel('time (years)')
        
def tipping_times():

	h = 0.05; loop = 40.0; T = 5000
	N_T = int(round(float(T)/h/loop))
	t = np.linspace(0, T, N_T)

	eta1 = 2.65; eta2 = 1.; eta3 = 0.3; RT = 1/200.
	sigT=0.0005; sigS=0.0005
	params = np.asarray([eta1,eta2,eta3,RT,sigT,sigS])

	start = 1000
	dur = 300
	ampl = 0.35
	ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur/h)),int((T-start-dur)/h)*[ampl]),)
	eta1_timeseries = np.concatenate((int(start/h/loop)*[eta1],np.linspace(eta1,eta1+ampl,int(dur/h/loop)),int((T-start-dur)/h/loop)*[eta1+ampl]),)

	basin_data = np.load('basin_data.npy')
	node_unstable = np.loadtxt('stommel_bif_unstable.txt',unpack=True)
	node_stable = np.loadtxt('stommel_bif_cold.txt',unpack=True)
	etagrid3 = np.arange(2.6, 3.05, 0.0025)
	spl = UnivariateSpline(node_unstable[0][::-1], node_unstable[1][::-1], s=0, k=1)
	unstable_intpT = spl(etagrid3)
	spl = UnivariateSpline(node_unstable[0][::-1], node_unstable[2][::-1], s=0, k=1)
	unstable_intpS = spl(etagrid3)
	spl = UnivariateSpline(node_stable[0], node_stable[1], s=0, k=1)
	stable_intpT = spl(etagrid3)
	spl = UnivariateSpline(node_stable[0], node_stable[2], s=0, k=1)
	stable_intpS = spl(etagrid3)
	
	idx = find_idx(etagrid3, 3.0)
	saddleT_fin = unstable_intpT[idx]
	saddleS_fin = unstable_intpS[idx]
	coldT_fin = stable_intpT[idx]
	coldS_fin = stable_intpS[idx]
	
	def calc_tipping_time_alt(t, s):
		idx = len(t)
		for i in range(int(1000/h/loop) + 6,len(t)):
			if (t[i-5]-s[i-5]>.1 and t[i-4]-s[i-4]>.1 and t[i-3]-s[i-3]>.1 and t[i-2]-s[i-2]>.1 and t[i-1]-s[i-1]>.1 and t[i]-s[i]>.1):
				idx = i - 5
			
				break
		
		return idx
		
	def calc_tipping_slice(t, s):
		idx1 = len(t); idx2 = len(t)
		start = 0
		for i in range(int(1000/h/loop) + 6,len(t)):
			if (t[i-5]-s[i-5]>.06 and t[i-4]-s[i-4]>.06 and t[i-3]-s[i-3]>.06 and t[i-2]-s[i-2]>.06 and t[i-1]-s[i-1]>.06 and t[i]-s[i]>.06 and start==0):
				idx1 = i - 5
				start=1
			if (t[i-5]-s[i-5]>.1 and t[i-4]-s[i-4]>.1 and t[i-3]-s[i-3]>.1 and t[i-2]-s[i-2]>.1 and t[i-1]-s[i-1]>.1 and t[i]-s[i]>.1):
				idx2 = i - 5
				break
		
		return idx1, idx2
			

	U_0 = [2.4,2.5]
	
	samples = 100000
	noise_vals = [0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002]
	
	start_time = time.time()
	fig=pl.figure(figsize=(5.,4.))
	pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
	
	for j in range(len(noise_vals)):
		tip=0
		tip_times=[]
		for i in range(samples):
			sigT=sigS=noise_vals[j]
			params = np.asarray([eta1,eta2,eta3,RT,sigT,sigS])
			Te0,S0 = sde_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
			if (S0[-1]<2.4 and S0[-2]<2.4 and S0[-3]<2.4 and S0[-4]<2.4):
				tip+=1
				basin_idx, crit_basin, saddleT, saddleS, fixT, fixS, Ttip, Stip = calc_basin_crossing(eta1_timeseries,Te0,S0, np.linspace(2.65,3.,120),basin_data,np.linspace(0.8,4.5,500), np.linspace(1.,4.,500), etagrid3, unstable_intpT, unstable_intpS, stable_intpT, stable_intpS)
				start_idx, tip_idx = calc_tipping_slice(Te0, S0)
				tip_times.append(tip_idx*h*loop)
				Te, S = Te0,S0
				print(tip_idx*h*loop)
				if tip_idx*h*loop>2000.:
					print('Late tipping')
					break
				
		xgrid = np.linspace(min(tip_times)-30,max(tip_times)+30,1000)
		kde = KernelDensity(kernel='gaussian', bandwidth=4.).fit(np.asarray(tip_times)[:, np.newaxis])
		log_dens = kde.score_samples(xgrid[:, np.newaxis])
		pl.plot(xgrid, np.exp(log_dens))
		print('Sigma = ', noise_vals[j])
		print('fraction of R-tippings: ',float(tip)/samples)
		print('average R-tipping time: ', np.mean(tip_times))
		print('St.Dev. R-tipping time: ', np.std(tip_times))
		

	pl.xlabel('Time of basin crossing'); pl.ylabel('PDF')
	print('simulation time (min)', (time.time()-start_time)/60.)
	

	T0_values = np.linspace(1.,4.,500); S0_values = np.linspace(0.8,4.5,500)
	fig=pl.figure(figsize=(5.,4.))
	pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
	pl.suptitle('Tipping time: %s'%(tip_idx*h*loop))
	crit_basin += -0.1
	pl.imshow(crit_basin,  cmap='RdYlBu', origin='lower',extent=(T0_values.min(),T0_values.max(),S0_values.min(),S0_values.max()),aspect="auto",alpha=0.9, vmin=-2.2, vmax=1.6)

	pl.plot(fixT,fixS,'o', color='tomato', markersize=10, markerfacecolor='none')
	pl.plot(saddleT,saddleS, '^', color='black', markersize=10, markerfacecolor='none')
	pl.plot(saddleT_fin,saddleS_fin,'^',color='black',markersize=10)
	pl.plot(coldT_fin, coldS_fin,'o',color='tomato',markersize=10)
	pl.plot(Te,S,color='black')
	pl.plot(Te[start_idx:tip_idx],S[start_idx:tip_idx],color='yellow')
	pl.plot(np.linspace(2.06,3.06, 100), np.linspace(2.,3., 100), ':', color='black')
	pl.plot(np.linspace(2.1,3.1, 100), np.linspace(2.,3., 100), ':', color='black')
	pl.xlim(2.1,3.1);pl.ylim(2.2,3.2)
	pl.xlabel('T'); pl.ylabel('S')
	
	fig=pl.figure(figsize=(6.,3.))
	pl.subplots_adjust(left=0.19, bottom=0.2, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
	pl.plot(t, Te, color='black')
	pl.plot(t[start_idx:tip_idx], Te[start_idx:tip_idx], color='yellow')
	pl.xlabel('Simulation time'); pl.ylabel('T')

	
def earlywarning_pullback_timeseries():

	h = 0.05; loop = 40.0; T = 4000
	N_T = int(round(float(T)/h/loop))
	t = np.linspace(0, T, N_T)

	eta1 = 2.65; eta2 = 1.; eta3 = 0.3; RT = 1/200.
	sigT=0.; sigS=0.
	params = np.asarray([eta1,eta2,eta3,RT,sigT,sigS])
	start = 1000
	dur = 300
	ampl = 0.35
	ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur/h)),int((T-start-dur)/h)*[ampl]),)

	U_0 = [2.4,2.5]
	
	### Consider deterministic trajectory as (approximation to) pullback attractor
	T_pullb,S_pullb = sde_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
	q_pullb = T_pullb-S_pullb
	sigT=0.001; sigS=0.001
	params = np.asarray([eta1,eta2,eta3,RT,sigT,sigS])
	
	def calc_tipping_time_alt(t, s):
		idx = len(t)
		for i in range(int(1000/h/loop) + 6,len(t)):
			if (t[i-5]-s[i-5]>.1 and t[i-4]-s[i-4]>.1 and t[i-3]-s[i-3]>.1 and t[i-2]-s[i-2]>.1 and t[i-1]-s[i-1]>.1 and t[i]-s[i]>.1):
				idx = i - 5
			
				break
		
		return idx
		
	def calc_tipping_slice(t, s):
		idx1 = len(t); idx2 = len(t)
		start = 0
		for i in range(int(1000/h/loop) + 6,len(t)):
			if (t[i-5]-s[i-5]>.06 and t[i-4]-s[i-4]>.06 and t[i-3]-s[i-3]>.06 and t[i-2]-s[i-2]>.06 and t[i-1]-s[i-1]>.06 and t[i]-s[i]>.06 and start==0):
				idx1 = i - 5
				start=1
			if (t[i-5]-s[i-5]>.1 and t[i-4]-s[i-4]>.1 and t[i-3]-s[i-3]>.1 and t[i-2]-s[i-2]>.1 and t[i-1]-s[i-1]>.1 and t[i]-s[i]>.1):
				idx2 = i - 5
				break
		
		return idx1, idx2

	### make sure that window and thin are commensurable (window/thin integer)
	lag = 1 # lag for autocorrelation
	thin = 5
	w = 200
	window = int(w/(h*loop))
	t_kernel_g = 400 # moving average kernel width in years
	kernel = int(t_kernel_g/loop/h)
	if kernel % 2 == 0:
		kernel += 1
	samples= 300000
	tip=0
	varT_all=[]; acT_all=[]; cc_all = []; T_all = []; T_all_filt = []; S_all = []; T_all_notip = []; S_all_notip = []; ews_all = []
	varT_all1=[]; acT_all1=[]
	varT_all2=[]; acT_all2=[]
	slices1 = []; splines1 = []; res1 = []
	slices2 = []; splines2 = []; res2 = []
	
	varT_aligned = []; acT_aligned = []; cc_aligned = []; ews_aligned = []
	T_aligned = []; T_slices = []; S_slices = []
	
	emb = 100
	start_time = time.time()
	tip_times = []
	s1 = 1900.; s2 = 2200.
	for i in range(samples):
		Te0,S0 = sde_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
		
		if (S0[-1]<2.4 and S0[-2]<2.4 and S0[-3]<2.4 and S0[-4]<2.4):
			tip+=1
			start_idx, tip_idx = calc_tipping_slice(Te0, S0)
			tip_times.append(tip_idx*h*loop)
			if (400.>tip_idx*h*loop-start_idx*h*loop and tip_idx*h*loop-start_idx*h*loop>300.):
				T_slices.append(Te0[start_idx:tip_idx])
				T_all.append(Te0)
				S_all.append(S0)
			
			ews_all.append(calc_ews_running(Te0, S0, thin, window))

			acT,acS,varT,varS,cc = calc_ews_spline(Te0,S0,window,thin,lag)
			varT_all.append(varT)
			acT_all.append(acT)
			cc_all.append(cc)
			'''
			acT1,acS1,varT1,varS1,cc1 = calc_ews_spline(Te0,S0,int(300/(h*loop)),thin,lag)
			varT_all1.append(varT1)
			acT_all1.append(acT1)
			acT2,acS2,varT2,varS2,cc2 = calc_ews_spline(Te0,S0,int(700/(h*loop)),thin,lag)
			varT_all2.append(varT2)
			acT_all2.append(acT2)
			'''
			
			
			T_all.append(Te0)
			S_all.append(S0)
			'''
			slice1 = Te0[int(s1/h/loop):int((s1+w)/h/loop)]
			model = np.polyfit(np.linspace(0,len(slice1),len(slice1)), slice1, 3)
			trend = np.polyval(model, np.linspace(0,len(slice1),len(slice1)))
			slices1.append(slice1)
			splines1.append(trend)
			res1.append(slice1-trend)
			
			slice2 = Te0[int(s2/h/loop):int((s2+w)/h/loop)]
			model = np.polyfit(np.linspace(0,len(slice2),len(slice2)), slice2, 3)
			trend = np.polyval(model, np.linspace(0,len(slice2),len(slice2)))
			slices2.append(slice2)
			splines2.append(trend)
			res2.append(slice2-trend)
			'''
		else:
			T_all_notip.append(Te0)
			S_all_notip.append(S0)
			
	print('simulation time (min)', (time.time()-start_time)/60.)
	start_time = time.time()

	print('fraction of R-tippings: ',float(tip)/samples)
	print('fraction of >100 year saddle approaches: ',float(len(T_slices))/samples)

	#np.save('jacobi_300_w250.npy', np.asarray([ews_all, ews_aligned]))

	mean_T,percentile_2_T,percentile_97_T = calc_mean_conf(T_all,len(Te0),tip)
	

	#mean_Tf,percentile_2_Tf,percentile_97_Tf = calc_mean_conf(T_all_filt,len(Te0),tip)

	mean_varT,percentile_2_varT,percentile_97_varT = calc_mean_conf(varT_all,len(varT),tip)
	mean_acT,percentile_2_acT,percentile_97_acT = calc_mean_conf(acT_all,len(varT),tip)
	mean_cc,percentile_2_cc,percentile_97_cc = calc_mean_conf(cc_all,len(varT),tip)
	
	#mean_varT1,percentile_2_varT1,percentile_97_varT1 = calc_mean_conf(varT_all1,len(varT1),tip)
	#mean_acT1,percentile_2_acT1,percentile_97_acT1 = calc_mean_conf(acT_all1,len(varT1),tip)
	
	#mean_varT2,percentile_2_varT2,percentile_97_varT2 = calc_mean_conf(varT_all2,len(varT2),tip)
	#mean_acT2,percentile_2_acT2,percentile_97_acT2 = calc_mean_conf(acT_all2,len(varT2),tip)

	mean_ews,percentile_2_ews,percentile_97_ews = calc_mean_conf(ews_all,len(ews_all[0]),tip)
	
	t_ews = t[window::thin]; t_ews = t_ews[:len(mean_ews)]
	
	### Ensemble variance and autocorrelation
	var_ens = [np.std([T_all[i][j] for i in range(len(T_all))])**2 for j in range(len(T_all[0]))]
	#ac_ens = [1./tip*np.sum([(T_all[i][j]-mean_T[j])*(T_all[i][j-1]-mean_T[j-1]) for i in range(len(T_all))]) for j in range(1,len(T_all[0]))]
	#ac_ens = [ac_ens[i]/var_ens[i] for i in range(len(ac_ens))]
	
	#np.save('ews_stommel_300_w200_jacn2.npy', np.asarray([mean_T,percentile_2_T,percentile_97_T, var_ens, varT_all, acT_all, ews_all]))
	
	### Factor to get correct Jacobian values
	fact = 1./h/RT/loop
	
	def test_statistic(T, S):
		#return ews.estimated_cc(T[10:], S[:-10])#10
		return fact*calc_jacobian_neighbors(T, S)
		#return ews.cy_fourth(S)
		#return ews.cy_skew(S)
		#return ews.cy_asym(T, 12)
		#return adfuller(S, regression='ct')[0]
		#return ews.cy_higher_ac(T)
		
	def ews_ensemble(slT, slS):
		var =[np.std(slT[i])**2 for i in range(len(slT))] 
		ac = [ews.estimated_ac1(slT[i],1) for i in range(len(slT))]
		#cc = [ews.estimated_cc(slT[i],slS[i]) for i in range(len(slT))]
		#ts = [test_statistic(slT[i],slS[i]) for i in range(len(slT))]
		return var, ac, var, ac#cc, ts
	
	saddle_times = [len(T_slices[i])*2. for i in range(len(T_slices))]
	

	T_slices_splined = [detrend_cubic(T_slices[i]) for i in range(len(T_slices))]
	S_slices_splined = [detrend_cubic(S_slices[i]) for i in range(len(T_slices))]
	cut = 1
	T_slices_splined = [T_slices_splined[i][cut:-cut] for i in range(len(T_slices))]
	S_slices_splined = [S_slices_splined[i][cut:-cut] for i in range(len(T_slices))]
	
	T_vals = [item for sublist in T_slices_splined for item in sublist]
	S_vals = [item for sublist in S_slices_splined for item in sublist]
	
	increments_T = [T_slices_splined[i][1:]-T_slices_splined[i][:-1] for i in range(len(T_slices))]
	increments_S = [S_slices_splined[i][1:]-S_slices_splined[i][:-1] for i in range(len(T_slices))]
	#increments_T = [item for sublist in increments_T for item in sublist]
	#increments_S = [item for sublist in increments_S for item in sublist]
	
	T_vals = [item for sublist in T_slices_splined for item in sublist]
	S_vals = [item for sublist in S_slices_splined for item in sublist]

	### Calculate equivalent Var, AC, CC for slices of the same length for on and off (before and after) attractors
	off_slicesT = [detrend_cubic(T_all[i][int(start/h/loop)-len(T_slices[i]):int(start/h/loop)]) for i in range(len(T_slices))]
	off_slicesS = [detrend_cubic(S_all[i][int(start/h/loop)-len(T_slices[i]):int(start/h/loop)]) for i in range(len(T_slices))]
	off_slicesT = [off_slicesT[i][cut:-cut] for i in range(len(T_slices))]
	off_slicesS = [off_slicesS[i][cut:-cut] for i in range(len(T_slices))]
	
	increments_T_off = [off_slicesT[i][1:]-off_slicesT[i][:-1] for i in range(len(T_slices))]
	increments_S_off = [off_slicesS[i][1:]-off_slicesS[i][:-1] for i in range(len(T_slices))]
	
	n_notip = min([len(T_all_notip), len(T_slices)])
	off_slicesT_aft = [detrend_cubic(T_all_notip[i][-len(T_slices[i]):]) for i in range(n_notip)]
	off_slicesS_aft = [detrend_cubic(S_all_notip[i][-len(T_slices[i]):]) for i in range(n_notip)]

	#increments_T_off_aft = [off_slicesT_aft[i][1:]-off_slicesT_aft[i][:-1] for i in range(n_notip)]
	#increments_S_off_aft = [off_slicesS_aft[i][1:]-off_slicesS_aft[i][:-1] for i in range(n_notip)]

	on_slicesT = [detrend_cubic(T_all[i][-len(T_slices[i]):]) for i in range(len(T_slices))]
	on_slicesS = [detrend_cubic(S_all[i][-len(T_slices[i]):]) for i in range(len(T_slices))]
	on_slicesT = [on_slicesT[i][cut:-cut] for i in range(len(T_slices))]
	on_slicesS = [on_slicesS[i][cut:-cut] for i in range(len(T_slices))]
	
	increments_T_on = [on_slicesT[i][1:]-on_slicesT[i][:-1] for i in range(len(T_slices))]
	increments_S_on = [on_slicesS[i][1:]-on_slicesS[i][:-1] for i in range(len(T_slices))]
	
	print('Calc. misc time (min)', (time.time()-start_time)/60.)
	start_time = time.time()

	#var_slices, ac_slices, cc_slices, ts_slices = ews_ensemble(T_slices_splined, S_slices_splined)##increments_T, increments_S
	#var_slices_off, ac_slices_off, cc_slices_off, ts_slices_off = ews_ensemble(off_slicesT, off_slicesS)##increments_T_off, increments_S_off
	#var_slices_off_aft, ac_slices_off_aft, cc_slices_off_aft, ts_slices_off_aft = ews_ensemble(off_slicesT_aft, off_slicesS_aft)##increments_T_off_aft, increments_S_off_aft
	#var_slices_on, ac_slices_on, cc_slices_on, ts_slices_on = ews_ensemble(on_slicesT, on_slicesS)##increments_T_on, increments_S_on

	print('Jacobian calc time (min)', (time.time()-start_time)/60.)
	
	fig=pl.figure(figsize=(7.,11.))
	pl.subplots_adjust(left=0.2, bottom=0.1, right=0.92, top=0.97, wspace=0.25, hspace=0.2)
	pl.suptitle('R-tipping (%s). %s y ramp. %s y window. %s y filter. Noise %s'%(float(tip)/samples,dur,w,t_kernel_g,sigT), fontsize=12)
	ax1 = pl.subplot(511)
	ax1.fill_between(t,percentile_2_T,percentile_97_T,alpha=0.2,label='95% confidence', color='black')
	ax1.plot(t,mean_T,label='T', color='black')
	ax1.plot(t,T_pullb, '--')
	pl.ylabel('T')
	ax2 = ax1.twinx()
	ax2.plot(np.linspace(0,T,len(ramp)),ramp,':', color='purple', label='ramp')
	pl.legend(loc='best'); pl.xlim(-30,T+30)
	
	#pl.subplot(512)
	#pl.fill_between(t,percentile_2_Tf,percentile_97_Tf,alpha=0.2,label='95% confidence', color='black')
	#pl.plot(t,mean_Tf, color='black')
	#pl.legend(loc='best'); pl.xlim(-30,T+30); pl.ylabel('T detr.')
	
	ax1=pl.subplot(513)
	pl.fill_between(t_ews,percentile_2_varT,percentile_97_varT,alpha=0.2,label='95% confidence', color='black')
	pl.plot(t_ews,mean_varT,label='var(T)', color='black')
	#t_ews1 = t[int(300/(h*loop))::thin]; t_ews1 = t_ews1[:len(mean_varT1)]
	#pl.plot(t_ews1,mean_varT1,label='300y', color='forestgreen')
	#t_ews2 = t[int(700/(h*loop))::thin]; t_ews2 = t_ews2[:len(mean_varT2)]
	#pl.plot(t_ews2,mean_varT2,label='700y', color='royalblue')
	ax2 = ax1.twinx()
	ax2.plot(t ,var_ens,  color='tomato', label='Ens.')
	pl.legend(loc='best')
	pl.xlim(-30,T+30); pl.ylabel('var(T)')
	
	ax1=pl.subplot(514)
	pl.fill_between(t_ews,percentile_2_acT,percentile_97_acT,alpha=0.2,label='95% confidence', color='black')
	pl.plot(t_ews,mean_acT,label='ac(T)', color='black')
	#ax2 = ax1.twinx()
	#pl.plot(t[101:] ,ac_ens[100:],  color='tomato', label='Ens.')
	pl.xlim(-30,T+30); pl.ylabel('ac(T)'); pl.xlabel('time (years)')
	
	pl.subplot(515)
	pl.fill_between(t_ews,percentile_2_ews,percentile_97_ews,alpha=0.2,label='95% confidence', color='black')
	pl.plot(t_ews,mean_ews, label='cc(S,T)', color='black')
	pl.xlim(-30,T+30); pl.ylabel('$\iota$'); pl.xlabel('time (years)')
	
	#pl.subplot(515)
	#pl.fill_between(t_ews,percentile_2_cc,percentile_97_cc,alpha=0.2,label='95% confidence', color='black')
	#pl.plot(t_ews,mean_cc,label='cc(S,T)', color='black')
	#pl.legend(loc='best'); pl.xlim(-30,T+30); pl.ylabel('cc(S,T)'); pl.xlabel('time (years)')

	'''
	fig=pl.figure(figsize=(8.,6.))
	pl.subplots_adjust(left=0.2, bottom=0.16, right=0.92, top=0.97, wspace=0.25, hspace=0.27)
	
	pl.subplot(221)
	
	xgrid, dens = calc_pdf(var_slices_on)
	pl.plot(xgrid, dens, '--',color='royalblue')
	xgrid, dens = calc_pdf(var_slices_off)
	pl.plot(xgrid, dens, color='black')
	xgrid, dens = calc_pdf(var_slices_off_aft)
	pl.plot(xgrid, dens, ':', color='purple')
	xgrid, dens = calc_pdf(var_slices)
	pl.plot(xgrid, dens, color='tomato')
	pl.xlabel('Var'); pl.ylabel('PDF')
	
	pl.subplot(222)
	
	xgrid, dens = calc_pdf(ac_slices_on)
	pl.plot(xgrid, dens, '--',color='royalblue')
	xgrid, dens = calc_pdf(ac_slices_off)
	pl.plot(xgrid, dens, color='black')
	xgrid, dens = calc_pdf(ac_slices_off_aft)
	pl.plot(xgrid, dens, ':', color='purple')
	xgrid, dens = calc_pdf(ac_slices)
	pl.plot(xgrid, dens, color='tomato')
	pl.xlabel('AC'); pl.ylabel('PDF')
	

	pl.subplot(223)
	
	xgrid, dens = calc_pdf(cc_slices_on)
	pl.plot(xgrid, dens, '--',color='royalblue')
	xgrid, dens = calc_pdf(cc_slices_off)
	pl.plot(xgrid, dens, color='black')
	xgrid, dens = calc_pdf(cc_slices_off_aft)
	pl.plot(xgrid, dens, ':', color='purple')
	xgrid, dens = calc_pdf(cc_slices)
	pl.plot(xgrid, dens, color='tomato')
	pl.xlabel('CC'); pl.ylabel('PDF')


	ax1=pl.subplot(224)
	
	xgrid, dens = calc_pdf(ts_slices_on)
	pl.plot(xgrid, dens, '--', color='royalblue')
	xgrid, dens = calc_pdf(ts_slices_off)
	pl.plot(xgrid, dens, color='black')
	xgrid, dens = calc_pdf(ts_slices_off_aft)
	pl.plot(xgrid, dens, ':', color='purple')
	xgrid, dens = calc_pdf(ts_slices)
	pl.plot(xgrid, dens, color='tomato')
	pl.xlabel('Jac(T,S)'); pl.ylabel('PDF')


	### test significance of EWS in the slices: randomly sample from saddle and 'off' distribution, and record whether EWS (saddle) > EWS (off)
	# or just do it with all combinations...
	significant = 0.
	for i in range(len(ts_slices)):
		for j in range(len(ts_slices)):
			if ts_slices[i]<ts_slices_off[j]:
				significant+=1.
				#print(significant)
	significant = significant/(len(ts_slices)**2)
	print(significant)
	
	#np.save('ews_jacobi_600_data.npy', np.asarray([ts_slices, ts_slices_off, ts_slices_on, ts_slices_off_aft]))

	
	eig_r1 = np.empty(len(T_slices)); eig_r2= np.empty(len(T_slices)); eig_i1= np.empty(len(T_slices)); eig_i2= np.empty(len(T_slices)); eig_r1_on= np.empty(len(T_slices))
	eig_r2_on= np.empty(len(T_slices)); eig_i1_on= np.empty(len(T_slices)); eig_i2_on= np.empty(len(T_slices)); eig_r1_off= np.empty(len(T_slices))
	eig_r2_off= np.empty(len(T_slices)); eig_i1_off= np.empty(len(T_slices)); eig_i2_off = np.empty(len(T_slices))
	

	for i in range(len(T_slices)):
		start_time = time.time()
		print(i, 'Data points in window: ', len(T_slices_splined[i]))
		er1, er2, ei1, ei2 = calc_jacobian_neighbors(T_slices_splined[i], S_slices_splined[i])#vector_ar1(T_slices_splined[i], S_slices_splined[i]) #calc_jacobian_neighbors#calc_cov_eigenvalues
		#eig_r1[i]= er1; eig_r2[i]= er2; eig_i1[i]= ei1; eig_i2[i]= ei2
		eig_r1[i]= fact*er1; eig_r2[i]= fact*er2; eig_i1[i]= fact*ei1; eig_i2[i]= fact*ei2
		#eig_i1[i] = er1/er2
		print(er1, er2, ei1, ei2)
		
		er1, er2, ei1, ei2 = calc_jacobian_neighbors(off_slicesT[i], off_slicesS[i])
		#eig_r1_off[i]= er1; eig_r2_off[i]= er2; eig_i1_off[i]= ei1; eig_i2_off[i]= ei2
		eig_r1_off[i]= fact*er1; eig_r2_off[i]= fact*er2; eig_i1_off[i]= fact*ei1; eig_i2_off[i]= fact*ei2
		print(er1, er2, ei1, ei2)
		#eig_i1_off[i]= er1/er2
		
		er1, er2, ei1, ei2 = calc_jacobian_neighbors(on_slicesT[i], on_slicesS[i])
		#eig_r1_on[i]= er1; eig_r2_on[i]= er2; eig_i1_on[i]= ei1; eig_i2_on[i]= ei2
		eig_r1_on[i]= fact*er1; eig_r2_on[i]= fact*er2; eig_i1_on[i]= fact*ei1; eig_i2_on[i]= fact*ei2
		print(er1, er2, ei1, ei2)
		#eig_i1_on[i]= er1/er2
		print('Jacobian calc time (min)', (time.time()-start_time)/60.)
		print('--------------------------------')
		
	def ews_power(saddle, off):
		significant = 0.
		for i in range(len(saddle)):
			for j in range(len(saddle)):
				if saddle[i]<off[j]:
					significant+=1.
		significant = significant/(len(saddle)**2)
		return significant
		
	print('Power Indicator 1: ', ews_power(eig_r1, eig_r1_off))
	print('Power Indicator 2: ', ews_power(eig_r2, eig_r2_off))
	print('Power Indicator 3: ', ews_power(eig_i1, eig_i1_off))
	print('Power Indicator 4: ', ews_power(eig_i2, eig_i2_off))
		

	fig=pl.figure()
	pl.subplot(221)
	pl.plot(eig_r1, [len(T_slices_splined[i]) for i in range(len(T_slices_splined))], 'o')
	pl.subplot(222)
	pl.plot(eig_r2, [len(T_slices_splined[i]) for i in range(len(T_slices_splined))], 'o')
	pl.subplot(223)
	pl.plot(eig_i1, [len(T_slices_splined[i]) for i in range(len(T_slices_splined))], 'o')
	pl.subplot(224)
	pl.plot(eig_i2, [len(T_slices_splined[i]) for i in range(len(T_slices_splined))], 'o')

	fig=pl.figure()
	pl.subplot(311)
	for i in range(len(T_slices)):
		pl.plot(np.linspace(0,len(T_slices_splined[i])*2,len(T_slices_splined[i])), T_slices_splined[i])
	pl.subplot(312)
	for i in range(len(T_slices)):
		pl.plot(np.linspace(0,len(T_slices_splined[i])*2,len(T_slices_splined[i])), on_slicesS[i])
	pl.subplot(313)
	for i in range(len(T_slices)):
		pl.plot(np.linspace(0,len(T_slices_splined[i])*2,len(T_slices_splined[i])), off_slicesS[i])
	
	fig=pl.figure()
	pl.subplot(221)
	xgrid, dens = calc_pdf(eig_r1_on)
	pl.plot(xgrid, dens, color='royalblue')
	pl.plot(100*[-3.47], np.linspace(0,max(dens), 100), ':', color='royalblue')#-0.706
	xgrid, dens = calc_pdf(eig_r1_off)
	pl.plot(xgrid, dens, color='black')
	pl.plot(100*[1.31], np.linspace(0,max(dens), 100), ':', color='black')#-0.796
	xgrid, dens = calc_pdf(eig_r1)
	pl.plot(xgrid, dens, color='tomato')
	pl.plot(100*[-3.89], np.linspace(0,max(dens), 100), ':', color='tomato')#0.699
	
	pl.subplot(222)
	xgrid, dens = calc_pdf(eig_r2_on)
	pl.plot(xgrid, dens, color='royalblue')
	pl.plot(100*[1.704], np.linspace(0,max(dens), 100), ':', color='royalblue')#-2.89
	xgrid, dens = calc_pdf(eig_r2_off)
	pl.plot(xgrid, dens, color='black')
	pl.plot(100*[-2.41], np.linspace(0,max(dens), 100), ':', color='black')#-0.796
	xgrid, dens = calc_pdf(eig_r2)
	pl.plot(xgrid, dens, color='tomato')
	pl.plot(100*[2.825], np.linspace(0,max(dens), 100), ':', color='tomato')#-2.185


	pl.subplot(223)
	xgrid, dens = calc_pdf(eig_i1_on)
	pl.plot(xgrid, dens, color='royalblue')
	pl.plot(100*[-0.94], np.linspace(0,max(dens), 100), ':', color='royalblue')#0.
	xgrid, dens = calc_pdf(eig_i1_off)
	pl.plot(xgrid, dens, color='black')
	pl.plot(100*[2.51], np.linspace(0,max(dens), 100), ':', color='black')#1.264
	xgrid, dens = calc_pdf(eig_i1)
	pl.plot(xgrid, dens, color='tomato')
	pl.plot(100*[-2.763], np.linspace(0,max(dens), 100), ':', color='tomato')#0.

	pl.subplot(224)
	xgrid, dens = calc_pdf(eig_i2_on)
	pl.plot(xgrid, dens, color='royalblue')
	pl.plot(100*[-0.127], np.linspace(0,max(dens), 100), ':', color='royalblue')#0.
	xgrid, dens = calc_pdf(eig_i2_off)
	pl.plot(xgrid, dens, color='black')
	pl.plot(100*[-2.905], np.linspace(0,max(dens), 100), ':', color='black')#-1.264
	xgrid, dens = calc_pdf(eig_i2)
	pl.plot(xgrid, dens, color='tomato')
	pl.plot(100*[2.401], np.linspace(0,max(dens), 100), ':', color='tomato')#0.

	
	### ROC curves.
	thresh = np.linspace(-0.2,0.2, 500)
	roc_tp = np.empty(len(thresh))
	roc_fp = np.empty(len(thresh))
	ts_sorted = np.sort(ts_slices)
	ts_sorted_off = np.sort(ts_slices_off)
	for i in range(len(thresh)):
		idx_on = np.searchsorted(ts_sorted, thresh[i])
		p_fp = float(idx_on)/len(ts_slices)
		roc_fp[i] = p_fp
		idx_off = np.searchsorted(ts_sorted_off, thresh[i])
		p_tp = float(idx_off)/len(ts_slices)
		roc_tp[i] = p_tp
		#print(idx_on, p_fp, idx_off, p_tp)
		
	fig=pl.figure()
	pl.plot(roc_fp, roc_tp)


	fig=pl.figure(figsize=(7.,5.))
	pl.subplots_adjust(left=0.2, bottom=0.16, right=0.92, top=0.97, wspace=0.25, hspace=0.2)

	mean_slice,p2_slice,p97_slice = calc_mean_conf(slices1,len(slice1),tip)
	mean_spline,p2_spline,p97_spline = calc_mean_conf(splines1,len(slice1),tip)
	mean_res,p2_res,p97_res = calc_mean_conf(res1,len(slice1),tip)
	pl.subplot(211)
	#pl.plot(np.linspace(s1, s1+w, len(slice1)), mean_slice)
	#pl.plot(np.linspace(s1, s1+w, len(slice1)), mean_spline)
	pl.fill_between(np.linspace(s1, s1+w, len(slice1)),p2_res,p97_res,alpha=0.2,label='95% confidence', color='black')
	pl.plot(np.linspace(s1, s1+w, len(slice1)), mean_res, color='black')
	pl.ylabel('Residual T')
	
	mean_slice,p2_slice,p97_slice = calc_mean_conf(slices2,len(slice2),tip)
	mean_spline,p2_spline,p97_spline = calc_mean_conf(splines2,len(slice2),tip)
	mean_res,p2_res,p97_res = calc_mean_conf(res2,len(slice2),tip)
	pl.subplot(212)
	#pl.plot(np.linspace(s2, s2+w, len(slice2)), mean_slice)
	#pl.plot(np.linspace(s2, s2+w, len(slice2)), mean_spline)
	pl.fill_between(np.linspace(s2, s2+w, len(slice1)),p2_res,p97_res,alpha=0.2,label='95% confidence', color='black')
	pl.plot(np.linspace(s2, s2+w, len(slice1)), mean_res, color='black')
	pl.xlabel('Simulation time');pl.ylabel('Residual T')


	#eig_r1, eig_r2, eig_i1, eig_i2 = calc_eigenvalue_all(T_slices_splined, S_slices_splined)
	#eig_r1_on, eig_r2_on, eig_i1_on, eig_i2_on = calc_eigenvalue_all(on_slicesT, on_slicesS)
	#eig_r1_off, eig_r2_off, eig_i1_off, eig_i2_off = calc_eigenvalue_all(off_slicesT, off_slicesS)
	


	ac_funcT= ac_function(increments_T)#T_slices_splined
	ac_funcT_off = ac_function(increments_T_off)#off_slicesT
	ac_funcT_on= ac_function(increments_T_on)#
	ac_funcT_off_aft = ac_function(increments_T_off_aft)#
	
	ac_funcS= ac_function(increments_S)#
	ac_funcS_off = ac_function(increments_S_off)#
	ac_funcS_on= ac_function(increments_S_on)#
	ac_funcS_off_aft = ac_function(increments_S_off_aft)#
	
	cc_func= cc_function(increments_T, increments_S)#
	cc_func_off = cc_function(increments_T_off, increments_S_off)#
	cc_func_on= cc_function(increments_T_on, increments_S_on)#
	cc_func_off_aft = cc_function(increments_T_off_aft, increments_S_off_aft)#
	
	asym_funcT= asym_function(increments_T)#
	asym_funcT_off = asym_function(increments_T_off)#
	asym_funcT_on= asym_function(increments_T_on)#
	asym_funcT_off_aft = asym_function(increments_T_off_aft)#
	

	fig=pl.figure()
	pl.subplot(221)
	pl.plot(range(25), ac_funcT_on, '--', color='royalblue')
	pl.plot(range(25), ac_funcT_off_aft, color='purple')
	pl.plot(range(25), ac_funcT_off, color='black')
	pl.plot(range(25), ac_funcT, color='tomato')
	pl.xlabel('Lag'); pl.ylabel('Autocorrelation (T)')
	
	pl.subplot(222)
	pl.plot(range(25), ac_funcS_on, '--', color='royalblue')
	pl.plot(range(25), ac_funcS_off_aft, color='purple')
	pl.plot(range(25), ac_funcS_off, color='black')
	pl.plot(range(25), ac_funcS, color='tomato')
	pl.xlabel('Lag'); pl.ylabel('Autocorrelation (S)')
	
	pl.subplot(223)
	pl.plot(range(-25,26), cc_func_on, '--', color='royalblue')
	pl.plot(range(-25,26), cc_func_off_aft, color='purple')
	pl.plot(range(-25,26), cc_func_off, color='black')
	pl.plot(range(-25,26), cc_func, color='tomato')
	pl.xlabel('Lag'); pl.ylabel('Crosscorrelation (T,S)')
	
	pl.subplot(224)
	pl.plot(range(25), asym_funcT_on, '--', color='royalblue')
	pl.plot(range(25), asym_funcT_off_aft, color='purple')
	pl.plot(range(25), asym_funcT_off, color='black')
	pl.plot(range(25), asym_funcT, color='tomato')
	pl.xlabel('Lag'); pl.ylabel('Asymmetry (T)')
	


	fig=pl.figure()
	pl.plot(eig_r1, eig_r2, 'o')
		

	### Compute correlation of subsequent increments or residuals
	alpha_vals = np.linspace(0, np.pi, 200)
	corr_vals = np.empty(len(alpha_vals))
	corr_vals_off = np.empty(len(alpha_vals))
	corr_vals_on = np.empty(len(alpha_vals))
	tau = 10 # lag of correlation
	for j in range(len(alpha_vals)):
		corrs = np.empty(len(increments_T)); corrs_off = np.empty(len(increments_T)); corrs_on = np.empty(len(increments_T))
		for i in range(len(increments_T)):
			#incT = np.asarray(increments_T[i]); incS = np.asarray(increments_S[i])
			incT = np.asarray(T_slices_splined[i]); incS = np.asarray(S_slices_splined[i])
			Z = np.sin(alpha_vals[j])*incT +np.cos(alpha_vals[j])*incS
			corrs[i] = pearsonr(Z[tau:],Z[:-tau])[0]
			#print(Z[0], len(Z))
			
			#incT = np.asarray(increments_T_off[i]); incS = np.asarray(increments_S_off[i])
			incT = np.asarray(off_slicesT[i]); incS = np.asarray(off_slicesS[i])
			Z = np.sin(alpha_vals[j])*incT +np.cos(alpha_vals[j])*incS
			corrs_off[i] = pearsonr(Z[tau:],Z[:-tau])[0]
			#print(Z[0], len(Z))
			
			#incT = np.asarray(increments_T_on[i]); incS = np.asarray(increments_S_on[i])
			incT = np.asarray(on_slicesT[i]); incS = np.asarray(on_slicesS[i])
			Z = np.sin(alpha_vals[j])*incT +np.cos(alpha_vals[j])*incS
			corrs_on[i] = pearsonr(Z[tau:],Z[:-tau])[0]
			
		#print('Average correlation of Increments: ', np.mean(corrs_off))
		corr_vals[j] = np.mean(corrs)
		corr_vals_off[j] = np.mean(corrs_off)
		corr_vals_on[j] = np.mean(corrs_on)

		
	fig=pl.figure()
	pl.plot(alpha_vals, corr_vals, color='tomato')
	pl.plot(alpha_vals, corr_vals_off, color='black')
	pl.plot(alpha_vals, corr_vals_on, '--', color='royalblue')
	'''
	
	'''
	fig=pl.figure()
	#for i in range(len(T_slices)):
	#	pl.plot(increments_T[i], increments_S[i], ',', color='black')
	pl.plot(T_vals, S_vals, ',', color='black')
	#pl.plot(increments_T, increments_S, ',', color='black')
	pl.plot(np.linspace(-0.05, 0.05, 100), np.linspace(-0.05*1.624, 0.05*1.624, 100), color='tomato')
	pl.plot(np.linspace(-0.05, 0.05, 100), np.linspace(-0.05*0.602, 0.05*0.602, 100), color='royalblue')
	'''
	
	#np.save('ews_on_off_saddle_300.npy',np.asarray([var_slices_on, ac_slices_on, cc_slices_on, var_slices_off, ac_slices_off, cc_slices_off, var_slices, ac_slices,cc_slices] ))
	'''

	t_ews_aligned = np.linspace(w, 2000, len(ews_aligned[0]))#2200	
	
	fig=pl.figure()
	pl.subplot(411)
	pl.fill_between(np.linspace(0,2000,1000),percentile_2_T_aligned,percentile_97_T_aligned,alpha=0.2,label='95% confidence', color='black')
	pl.plot(np.linspace(0,2000,1000),mean_T_aligned, color='black'); pl.xlim(0,2000)
	pl.subplot(412)
	pl.fill_between(t_ews_aligned,percentile_2_ews_aligned,percentile_97_ews_aligned,alpha=0.2,label='95% confidence', color='black')
	pl.plot(t_ews_aligned,mean_ews_aligned, color='black'); pl.xlim(0,2000)
	pl.subplot(413)
	pl.fill_between(t,percentile_2_T,percentile_97_T,alpha=0.2,label='95% confidence', color='black')
	pl.plot(t,mean_T,label='T', color='black'); pl.xlim(-30,T+30)
	pl.subplot(414)
	pl.fill_between(t_ews,percentile_2_ews,percentile_97_ews,alpha=0.2,label='95% confidence', color='black')
	pl.plot(t_ews,mean_ews, color='black')
	pl.legend(loc='best'); pl.xlim(-30,T+30)#; pl.ylabel('T detr.')
	

	fig=pl.figure()
	ax1=pl.subplot(111)
	xgrid = np.linspace(min(eig)-.00001, max(eig)+.00001,1000)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(eig)/20.).fit(np.asarray(eig)[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.plot(xgrid, np.exp(log_dens), color='tomato')
	

	eig_ts = [ews_aligned[i][-1] for i in range(len(ews_aligned))]
	
	xgrid = np.linspace(min(eig_ts)-.001, max(eig_ts)+.001,1000)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(eig_ts)/20.).fit(np.asarray(eig_ts)[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	ax2 = ax1.twinx()
	ax2.plot(xgrid, np.exp(log_dens), '--', color='forestgreen')
	'''
	
	'''
	
	fig=pl.figure()
	pl.subplot(221)
	for i in range(len(T_slices)):
		pl.plot(len(T_slices[i])*2, var_slices[i] , 'o', color='black')
	pl.xlabel('Time spent near saddle'); pl.ylabel('Variance')
	pl.subplot(222)
	for i in range(len(T_slices)):
		pl.plot(len(T_slices[i])*2, ac_slices[i] , 'o', color='black')
	pl.xlabel('Time spent near saddle'); pl.ylabel('AC(1)')
	pl.subplot(223)
	for i in range(len(T_slices)):
		pl.plot(len(T_slices[i])*2, cc_slices[i] , 'o', color='black')
	pl.xlabel('Time spent near saddle'); pl.ylabel('CC')
	pl.subplot(224)
	#for i in range(len(T_slices)):
	#	pl.plot(len(T_slices[i])*2, eig_r1[i] , 'o', color='black')
	#pl.xlabel('Time spent near saddle'); pl.ylabel('lambda1')
	
	fig=pl.figure(figsize=(5.,4.))
	pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
	xgrid = np.linspace(min(saddle_times)-30,max(saddle_times)+30,1000)
	kde = KernelDensity(kernel='gaussian', bandwidth=4.).fit(np.asarray(saddle_times)[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	pl.plot(xgrid, np.exp(log_dens))#, color='black', alpha=0.2+j*0.1)
	pl.xlabel('Time spent near saddle'); pl.ylabel('PDF')
	
	

	
	#pl.subplot(313)
	#for i in range(len(T_slices)):
	#	pl.plot(t_ews_aligned, ews_aligned[i])



	fig=pl.figure()
	pl.subplot(311)
	for i in range(len(T_slices)):
		#pl.plot(np.linspace(0,len(T_slices[i])*2,len(T_slices[i])), T_slices[i])
		#pl.plot(np.linspace(0,len(T_slices[i])*2,len(T_slices[i])), S_slices_splined[i])
		pl.plot(np.linspace(0,len(increments_T[i])*2,len(increments_T[i])), increments_S[i])
	pl.subplot(312)
	for i in range(len(T_slices)):
		#pl.plot(np.linspace(0,len(T_slices[i])*2,len(T_slices[i])), on_slicesS[i])
		pl.plot(np.linspace(0,len(increments_T[i])*2,len(increments_T[i])), increments_S_on[i])
	pl.subplot(313)
	for i in range(len(T_slices)):
		#pl.plot(np.linspace(0,len(T_slices[i])*2,len(T_slices[i])), off_slicesS[i])
		pl.plot(np.linspace(0,len(increments_T[i])*2,len(increments_T[i])), increments_S_off[i])

	fig=pl.figure()
	pl.subplot(411)
	#for i in range(tip):
	#	pl.plot(np.linspace(0,2200,1200), T_aligned[i])
	pl.fill_between(np.linspace(0,2200,1200),percentile_2_T_aligned,percentile_97_T_aligned,alpha=0.2,label='95% confidence', color='black')
	pl.plot(np.linspace(0,2200,1200),mean_T_aligned, color='black')
	
	#pl.subplot(412)	
	#pl.fill_between(t_ews_aligned,percentile_2_varT_aligned,percentile_97_varT_aligned,alpha=0.2,label='95% confidence', color='black')
	#pl.plot(t_ews_aligned,mean_varT_aligned, color='black')
	
	#pl.subplot(413)	
	#pl.fill_between(t_ews_aligned,percentile_2_acT_aligned,percentile_97_acT_aligned,alpha=0.2,label='95% confidence', color='black')
	#pl.plot(t_ews_aligned,mean_acT_aligned, color='black')
	
	pl.subplot(414)	
	pl.fill_between(t_ews_aligned,percentile_2_ews_aligned,percentile_97_ews_aligned,alpha=0.2,label='95% confidence', color='black')
	pl.plot(t_ews_aligned,mean_ews_aligned, color='black')

	

	
	fig=pl.figure()
	pl.plot(t, Te0, color='black')
	pl.plot(t, smoothing_filter_gauss(Te0,kernel,4*kernel+1), '--', color='tomato')
	Te0 = T_all[0]
	pl.plot(t, Te0, color='black')
	pl.plot(t, smoothing_filter_gauss(Te0,kernel,4*kernel+1), '--', color='tomato')
	
	#pl.plot(t, T_pullb, '--', color='forestgreen')

	'''
	
def earlywarning():

        h = 0.05; loop = 40.0; T = 5000
        N_T = int(round(float(T)/h/loop))
        t = np.linspace(0, T, N_T)

        eta1 = 2.65; eta2 = 1.; eta3 = 0.3; RT = 1/200.
        sigT=0.0015; sigS=0.0015
        params = np.asarray([eta1,eta2,eta3,RT,sigT,sigS])

        start = 1000
        dur = 300
        ampl = 0.35
        ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur/h)),int((T-start-dur)/h)*[ampl]),)

        eta1_timeseries = np.concatenate((int(start/h/loop)*[eta1],np.linspace(eta1,eta1+ampl,int(dur/h/loop)),int((T-start-dur)/h/loop)*[eta1+ampl]),)

        node_unstable = np.loadtxt('stommel_bif_unstable.txt',unpack=True)
        print(node_unstable[0][::-1])

        node_stable = np.loadtxt('stommel_bif_cold.txt',unpack=True)
        print(node_stable[0])

        etagrid3 = np.arange(2.6, 3.05, 0.0025)
        spl = UnivariateSpline(node_unstable[0][::-1], node_unstable[1][::-1], s=0, k=1)
        unstable_intpT = spl(etagrid3)
        spl = UnivariateSpline(node_unstable[0][::-1], node_unstable[2][::-1], s=0, k=1)
        unstable_intpS = spl(etagrid3)

        spl = UnivariateSpline(node_stable[0], node_stable[1], s=0, k=1)
        stable_intpT = spl(etagrid3)
        spl = UnivariateSpline(node_stable[0], node_stable[2], s=0, k=1)
        stable_intpS = spl(etagrid3)

        idx = find_idx(etagrid3, 3.0)
        saddleT_fin = unstable_intpT[idx]
        saddleS_fin = unstable_intpS[idx]

        basin_data = np.load('basin_data.npy')

        U_0 = [2.4,2.5]

        lag = 1 # lag for autocorrelation
        thin = 10
        filt = 0.5/50 # Filter cut-off: fraction of 0.5 = 1/timestep = 1/(h*loop)= 1/2y
        w = 100
        window = int(w/(h*loop))

        t_kernel_g = 50 # moving average kernel width in years

        kernel = int(t_kernel_g/loop/h)
        if kernel % 2 == 0:
                kernel += 1

        start_time = time.time()

        samples=100
        tip=0
        varT_all=[];varS_all=[];acT_all=[];acS_all=[];cc_all=[];skew_all=[];crit_times=[];tip_times=[]
        varT_a_all=[];varS_a_all=[];acT_a_all=[];acS_a_all=[];cc_a_all=[];skew_a_all=[];asym_a_all=[];adf_a_all=[];ac2_a_all=[]
        T_align = []; S_align = []; T_all = []; T_all_filt = []
        for i in range(samples):
                        
                Te0,S0 = sde_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)

                Te0_detr = Te0 - smoothing_filter_gauss(Te0,kernel,4*kernel+1)
                S0_detr = S0 - smoothing_filter_gauss(S0,kernel,4*kernel+1)
                
                acT,acS,varT,varS,cc,skew,asym,adf,ac2 = calc_ews_filtered(Te0_detr,S0_detr,int(window),thin)

                if (Te0[-1]>S0[-1] and Te0[-2]>S0[-2] and Te0[-3]>S0[-3] and Te0[-4]>S0[-4]):
                        tip+=1
                        zero_idx = (np.abs((Te0-S0)-0.4333)).argmin()
                        crit_times.append(t[zero_idx])
                        varT_all.append(varT)
                        varS_all.append(varS)
                        acT_all.append(acT)
                        acS_all.append(acS)
                        cc_all.append(cc)
                        skew_all.append(skew)
                        T_all.append(Te0)
                        T_all_filt.append(Te0_detr)

                        tip_idx, crit_basin, saddleT, saddleS, fixT, fixS, Ttip, Stip = calc_basin_crossing(eta1_timeseries,Te0,S0, np.linspace(2.65,3.,120),basin_data,np.linspace(0.8,4.5,500), np.linspace(1.,4.,500), etagrid3, unstable_intpT, unstable_intpS, stable_intpT, stable_intpS)
                        print(tip_idx*h*loop)
                        tip_times.append(tip_idx*h*loop)
                        Ta = Te0_detr[tip_idx-400:tip_idx]
                        Sa = S0_detr[tip_idx-400:tip_idx]
                        T_align.append(Ta)
                        S_align.append(Sa)
                        acT_a,acS_a,varT_a,varS_a,cc_a,skew_a,asym_a,adf_a,ac2_a = calc_ews(Ta,Sa,int(window),5)
                        varT_a_all.append(varT_a)
                        varS_a_all.append(varS_a)
                        acT_a_all.append(acT_a)
                        acS_a_all.append(acS_a)
                        cc_a_all.append(cc_a)
                        skew_a_all.append(skew_a)
                        asym_a_all.append(asym_a)
                        adf_a_all.append(adf_a)
                        ac2_a_all.append(ac2_a)
                        

        mean_T,percentile_2_T,percentile_97_T = calc_mean_conf(T_all,len(Te0),tip)
        mean_Tf,percentile_2_Tf,percentile_97_Tf = calc_mean_conf(T_all_filt,len(Te0),tip)
        mean_varT,percentile_2_varT,percentile_97_varT = calc_mean_conf(varT_all,len(varT),tip)
        mean_varS,percentile_2_varS,percentile_97_varS = calc_mean_conf(varS_all,len(varT),tip)
        mean_acT,percentile_2_acT,percentile_97_acT = calc_mean_conf(acT_all,len(varT),tip)
        mean_acS,percentile_2_acS,percentile_97_acS = calc_mean_conf(acS_all,len(varT),tip)
        mean_cc,percentile_2_cc,percentile_97_cc = calc_mean_conf(cc_all,len(varT),tip)

        print('fraction of R-tippings: ',float(tip)/samples)
        t_ews = t[window::thin]

        print('average R-tipping time: ', np.mean(tip_times))
        print('St.Dev. R-tipping time: ', np.std(tip_times))

        ### PCA of (T,S)-fluctuations before and during R-tipping
        n_pca = 30
        T_bef = np.empty(n_pca*tip); S_bef = np.empty(n_pca*tip); T_aft = np.empty(n_pca*tip); S_aft = np.empty(n_pca*tip)
        fig = pl.figure()
        pl.subplot(121)
        for i in range(tip):
                pl.plot(T_align[i][:n_pca],S_align[i][:n_pca],',')
                T_bef[i*n_pca:(i+1)*n_pca] = T_align[i][:n_pca]
                S_bef[i*n_pca:(i+1)*n_pca] = S_align[i][:n_pca]
        pl.subplot(122)
        for i in range(tip):
                pl.plot(T_align[i][-n_pca:],S_align[i][-n_pca:],',')
                T_aft[i*n_pca:(i+1)*n_pca] = T_align[i][-n_pca:]
                S_aft[i*n_pca:(i+1)*n_pca] = S_align[i][-n_pca:]

        xgrid_befT = np.linspace(min(T_bef)-0.005,max(T_bef)+0.005,500)
        kde = KernelDensity(kernel='gaussian', bandwidth=.0005).fit(np.asarray(T_bef)[:, np.newaxis])
        log_dens_befT = kde.score_samples(xgrid_befT[:, np.newaxis])

        xgrid_aftT = np.linspace(min(T_aft)-0.005,max(T_aft)+0.005,500)
        kde = KernelDensity(kernel='gaussian', bandwidth=.0005).fit(np.asarray(T_aft)[:, np.newaxis])
        log_dens_aftT = kde.score_samples(xgrid_aftT[:, np.newaxis])

        xgrid_befS = np.linspace(min(S_bef)-0.005,max(S_bef)+0.005,500)
        kde = KernelDensity(kernel='gaussian', bandwidth=.0005).fit(np.asarray(S_bef)[:, np.newaxis])
        log_dens_befS = kde.score_samples(xgrid_befS[:, np.newaxis])

        xgrid_aftS = np.linspace(min(S_aft)-0.005,max(S_aft)+0.005,500)
        kde = KernelDensity(kernel='gaussian', bandwidth=.0005).fit(np.asarray(S_aft)[:, np.newaxis])
        log_dens_aftS = kde.score_samples(xgrid_aftS[:, np.newaxis])

        fig1=pl.figure()
        pl.subplot(121)
        pl.plot(xgrid_befT, np.exp(log_dens_befT),label='before')
        pl.plot(xgrid_aftT, np.exp(log_dens_aftT),label='after')
        pl.subplot(122)
        pl.plot(xgrid_befS, np.exp(log_dens_befS),label='before')
        pl.plot(xgrid_aftS, np.exp(log_dens_aftS),label='after')

        fig2=pl.figure()
        ax1 = fig2.add_subplot(221); ax1.set_title('normal Q-Q plot T before')
        sm.qqplot(T_bef,line='s',fit='False',ax=ax1)
        
        ax2 = fig2.add_subplot(222)
        ax2.set_title('normal Q-Q plot T after'); sm.qqplot(T_aft,line='s',fit='False',ax=ax2)

        ax3 = fig2.add_subplot(223); ax3.set_title('normal Q-Q plot S before')
        sm.qqplot(S_bef,line='s',fit='False',ax=ax3)

        ax4 = fig2.add_subplot(224); ax4.set_title('normal Q-Q plot S after')
        sm.qqplot(S_aft,line='s',fit='False',ax=ax4)


        '''
        ratio_bef = np.empty(tip); ratio_aft = np.empty(tip)
        for i in range(tip):
                Xp = np.asarray([T_align[i][:n_pca],S_align[i][:n_pca]])
                Xp = Xp.transpose()
                pca = PCA(n_components=2)
                pca.fit(Xp)
                ratio_bef[i] = pca.explained_variance_ratio_[0]/pca.explained_variance_ratio_[1]
                Xp = np.asarray([T_align[i][-n_pca:],S_align[i][-n_pca:]])
                Xp = Xp.transpose()
                pca = PCA(n_components=2)
                pca.fit(Xp)
                ratio_aft[i] = pca.explained_variance_ratio_[0]/pca.explained_variance_ratio_[1]


        xgrid_bef = np.linspace(min(ratio_bef)-0.1,max(ratio_bef)+0.1,500)
        kde = KernelDensity(kernel='gaussian', bandwidth=.1).fit(np.asarray(ratio_bef)[:, np.newaxis])
        log_dens_bef = kde.score_samples(xgrid_bef[:, np.newaxis])

        xgrid_aft = np.linspace(min(ratio_aft)-0.1,max(ratio_aft)+0.1,500)
        kde = KernelDensity(kernel='gaussian', bandwidth=.1).fit(np.asarray(ratio_aft)[:, np.newaxis])
        log_dens_aft = kde.score_samples(xgrid_aft[:, np.newaxis])

        fig1=pl.figure()
        pl.plot(xgrid_bef, np.exp(log_dens_bef),label='before')
        pl.plot(xgrid_aft, np.exp(log_dens_aft),label='after')
        pl.legend(loc='best')

                
        Xp = np.asarray([T_bef,S_bef])
        Xp = Xp.transpose()#; Xp = StandardScaler().fit_transform(Xp)
        pca = PCA(n_components=2)
        pca.fit(Xp)
        expl_var = pca.explained_variance_ratio_
        score =  pca.fit_transform(Xp)
        coeff = pca.components_

        print('Expl variance before: ', pca.explained_variance_ratio_)
        print('Components before: ', pca.components_)
        biplot(score,coeff,1,2)

        Xp = np.asarray([T_aft,S_aft])
        Xp = Xp.transpose()#; Xp = StandardScaler().fit_transform(Xp)
        pca = PCA(n_components=2)
        pca.fit(Xp)
        expl_var = pca.explained_variance_ratio_
        score =  pca.fit_transform(Xp)
        coeff = pca.components_

        print('Expl variance after: ', pca.explained_variance_ratio_)
        print('Components after: ', pca.components_)
        biplot(score,coeff,1,2)
        '''

        fig1=pl.figure()
        pl.suptitle('$\sigma_T=\sigma_S = %s$'%sigT)
        pl.plot(xgrid, np.exp(log_dens))
        pl.ylim(0,0.055);pl.xlim(1200,1700)

        T0_values = np.linspace(1.,4.,500); S0_values = np.linspace(0.8,4.5,500)
        fig=pl.figure(figsize=(5.,4.))
        pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
        pl.suptitle('Tipping time: %s'%(tip_idx*h*loop))
        crit_basin += -0.1
        pl.imshow(crit_basin,  cmap='RdYlBu', origin='lower',extent=(T0_values.min(),T0_values.max(),S0_values.min(),S0_values.max()),aspect="auto",alpha=0.9, vmin=-2.2, vmax=1.6)

        pl.plot(fixT,fixS,'o', color='tomato', markersize=10)
        pl.plot(Ttip,Stip,'x',color='black',markersize=10.)
        pl.plot(saddleT,saddleS, '^', color='gold', markersize=10)
        pl.plot(saddleT_fin,saddleS_fin,'+',color='black',markersize=7.)
        pl.plot(Te0,S0,color='forestgreen')
        pl.plot(Te0[:tip_idx],S0[:tip_idx],color='purple')
        pl.xlim(2.1,3.1);pl.ylim(2.2,3.2)
        pl.xlabel('T'); pl.ylabel('S')
        
        fig2=pl.figure()
        pl.suptitle('Stommel model: Time series EWS. R-tipping (%s). %s y ramp. %s y window. %s y filter. Noise %s'%(float(tip)/samples,dur,w,t_kernel_g,sigT))
        ax1 = pl.subplot(711)
        ax1.fill_between(t,percentile_2_T,percentile_97_T,alpha=0.2,label='95% confidence')
        ax1.plot(t,mean_T,label='T')
        ax1.plot(100*[np.mean(tip_times)], np.linspace(min(percentile_2_T),max(percentile_97_T),100),'--',color='black')
        ax2 = ax1.twinx()
        ax2.plot(np.linspace(0,T,len(ramp)),ramp,':',label='ramp')
        pl.legend(loc='best')
        pl.xlim(-30,T+30)
        pl.subplot(712)
        pl.fill_between(t,percentile_2_Tf,percentile_97_Tf,alpha=0.2,label='95% confidence')
        pl.plot(t,mean_Tf)
        pl.plot(100*[np.mean(tip_times)], np.linspace(min(percentile_2_Tf),max(percentile_97_Tf),100),'--',color='black')
        pl.legend(loc='best')
        pl.xlim(-30,T+30)       
        pl.subplot(713)
        pl.fill_between(t_ews,percentile_2_varT,percentile_97_varT,alpha=0.2,label='95% confidence')
        pl.plot(t_ews,mean_varT,label='var(T)')
        pl.plot(t_ews,mean_varS,label='var(S)')
        pl.legend(loc='best')
        pl.xlim(-30,T+30)
        pl.ylabel('var(T)')
        pl.subplot(714)
        pl.fill_between(t_ews,percentile_2_varS,percentile_97_varS,alpha=0.2,label='95% confidence')
        pl.plot(t_ews,mean_varS,label='var(S)')
        pl.legend(loc='best')
        pl.xlim(-30,T+30)
        pl.ylabel('var(S)')
        pl.subplot(715)
        pl.fill_between(t_ews,percentile_2_acT,percentile_97_acT,alpha=0.2,label='95% confidence')
        pl.plot(t_ews,mean_acT,label='ac(T)')
        pl.plot(t_ews,mean_acS,label='ac(S)')
        pl.legend(loc='best')
        pl.xlim(-30,T+30)
        pl.ylabel('ac(T)')
        pl.subplot(716)
        pl.fill_between(t_ews,percentile_2_acS,percentile_97_acS,alpha=0.2,label='95% confidence')
        pl.plot(t_ews,mean_acS,label='ac(S)')
        pl.legend(loc='best')
        pl.xlim(-30,T+30)
        pl.ylabel('ac(S)')
        pl.subplot(717)
        pl.fill_between(t_ews,percentile_2_cc,percentile_97_cc,alpha=0.2,label='95% confidence')
        pl.plot(t_ews,mean_cc,label='cc(T,S)')
        pl.legend(loc='best')
        pl.xlim(-30,T+30)
        pl.ylabel('cc(T,S)')
        pl.xlabel('time (years)')


        mean_T,percentile_2_T,percentile_97_T = calc_mean_conf(T_align,len(Ta),tip)
        mean_varT_a,percentile_2_varT_a,percentile_97_varT_a = calc_mean_conf(varT_a_all,len(varT_a),tip)
        mean_varS_a,percentile_2_varS_a,percentile_97_varS_a = calc_mean_conf(varS_a_all,len(varT_a),tip)
        mean_acT_a,percentile_2_acT_a,percentile_97_acT_a = calc_mean_conf(acT_a_all,len(varT_a),tip)
        mean_acS_a,percentile_2_acS_a,percentile_97_acS_a = calc_mean_conf(acS_a_all,len(varT_a),tip)
        mean_cc_a,percentile_2_cc_a,percentile_97_cc_a = calc_mean_conf(cc_a_all,len(varT_a),tip)
        mean_skew_a,percentile_2_skew_a,percentile_97_skew_a = calc_mean_conf(skew_a_all,len(varT_a),tip)
        mean_asym_a,percentile_2_asym_a,percentile_97_asym_a = calc_mean_conf(asym_a_all,len(varT_a),tip)
        mean_adf_a,percentile_2_adf_a,percentile_97_adf_a = calc_mean_conf(adf_a_all,len(varT_a),tip)
        mean_ac2_a,percentile_2_ac2_a,percentile_97_ac2_a = calc_mean_conf(ac2_a_all,len(varT_a),tip)

        t_ews_align = np.linspace(w,800,len(percentile_2_varT_a))

        mean97 = np.mean(percentile_97_cc_a[:40])
        mean2 = np.mean(percentile_2_cc_a[:40])

        print(len(t_ews_align))
        print(len(percentile_2_varT_a))

        print('simulation time (min)', (time.time()-start_time)/60.)

        fig2=pl.figure()
        pl.suptitle('Stommel model: Aligned time series EWS. R-tipping (%s). %s y ramp. %s y window. %s y filter. Noise %s'%(float(tip)/samples,dur,w,t_kernel_g,sigT))
        pl.subplot(811)
        pl.fill_between(np.linspace(0,800,len(Ta)),percentile_2_T,percentile_97_T,alpha=0.2)
        pl.plot(np.linspace(0,800,len(Ta)),mean_T,label='T detrended')
        pl.subplot(812)
        pl.fill_between(t_ews_align,percentile_2_varT_a,percentile_97_varT_a,alpha=0.2,label='95% confidence')
        pl.plot(t_ews_align,mean_varT_a,label='var(T)')
        pl.plot(t_ews_align,mean_varS_a,label='var(S)')
        pl.legend(loc='best')
        pl.ylabel('var(T)')
        pl.subplot(813)
        pl.fill_between(t_ews_align,percentile_2_acT_a,percentile_97_acT_a,alpha=0.2,label='95% confidence')
        pl.plot(t_ews_align,mean_acT_a,label='ac(T)')
        pl.plot(t_ews_align,mean_acS_a,label='ac(S)')
        pl.legend(loc='best')
        pl.ylabel('ac(T)')
        pl.subplot(814)
        pl.fill_between(t_ews_align,percentile_2_cc_a,percentile_97_cc_a,alpha=0.2,label='95% confidence')
        pl.plot(t_ews_align,len(t_ews_align)*[mean97],color='gray')     
        pl.plot(t_ews_align,len(t_ews_align)*[mean2],color='gray')           
        pl.plot(t_ews_align,mean_cc_a,label='cc(T,S)')
        pl.ylabel('cc(T,S)')
        pl.subplot(815)
        pl.fill_between(t_ews_align,percentile_2_skew_a,percentile_97_skew_a,alpha=0.2)
        pl.plot(t_ews_align,mean_skew_a)
        pl.ylabel('Skewness(T)')
        pl.subplot(816)
        pl.fill_between(t_ews_align,percentile_2_asym_a,percentile_97_asym_a,alpha=0.2)
        pl.plot(t_ews_align,mean_asym_a)
        pl.ylabel('Asymmetry(T)')
        pl.subplot(817)
        pl.fill_between(t_ews_align,percentile_2_adf_a,percentile_97_adf_a,alpha=0.2)
        pl.plot(t_ews_align,mean_adf_a)
        pl.ylabel('ADF stat. (T)')
        pl.subplot(818)
        pl.fill_between(t_ews_align,percentile_2_ac2_a,percentile_97_ac2_a,alpha=0.2)
        pl.plot(t_ews_align,mean_ac2_a)
        pl.ylabel('2nd order AC (T)')
        pl.xlabel('time (years)')
        
        
def trans_prob():

	h = 0.05; loop = 40.0; T = 5000
	N_T = int(round(float(T)/h/loop))
	t = np.linspace(0, T, N_T)

	eta1 = 2.65; eta2 = 1.; eta3 = 0.3; RT = 1/200.
	sigT=sigS=0.0002
	params = np.asarray([eta1,eta2,eta3,RT,sigT,sigS])

	start = 1000
	dur = 300
	ampl = 0.35
	ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur/h)),int((T-start-dur)/h)*[ampl]),)

	U_0 = [2.4,2.5]

	dur_vals = np.linspace(100,600,150)
	
	noise_vals = [0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002]
	
	samples = 1000
	
	start_time = time.time()
	fig=pl.figure()
	probs_all = []
	for k in range(len(noise_vals)):
		probs = np.empty(len(dur_vals))
		sigT=sigS = noise_vals[k]
		params = np.asarray([eta1,eta2,eta3,RT,sigT,sigS])
		for i in range(len(dur_vals)):
			tip = 0
			ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur_vals[i]/h)),int(2+(T-start-dur_vals[i])/h)*[ampl]),)
			for j in range(samples):
				Te0,S0 = sde_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
				if (Te0[-1]>S0[-1] and Te0[-2]>S0[-2] and Te0[-3]>S0[-3] and Te0[-4]>S0[-4]): 
					tip+=1.
			print(tip/samples)
			probs[i] = tip/samples
		
		pl.plot(dur_vals, probs)
		probs_all.append(probs)
	
	print('simulation time (min)', (time.time()-start_time)/60.)
	np.save('stommel_rtip_probs_noise_6.npy', np.asarray(probs_all))


def calc_basin_crossing(eta1_all,T_all,S_all,eta1_vals,basin_all,S_grid,T_grid, etagrid3, unstableT,unstableS, stableT, stableS):

        basin_all = basin_all[0]
        indi=0
        for i in range(len(T_all)):        
                S=S_all[i];T=T_all[i];eta1=eta1_all[i]
                idx = find_idx(eta1_vals, eta1)
                eta1_comp = eta1_vals[idx]

                basin = basin_all[idx]
                Tidx = find_idx(T_grid,T); Sidx = find_idx(S_grid,S)
                if basin[Sidx][Tidx]>0:
                        break
                        indi+=1
                if (indi>0 and basin[Sidx][Tidx]<0):
                        indi = 0

                if indi>3:
                        break
        idx_saddle = find_idx(etagrid3, eta1)
        saddleT = unstableT[idx_saddle]; saddleS = unstableS[idx_saddle]
        fixT = stableT[idx_saddle]; fixS = stableS[idx_saddle]

        return i, basin, saddleT, saddleS, fixT, fixS, T, S
        
        
def plot_realizations():

	node_warm = np.loadtxt('stommel_bif_warm.txt',unpack=True)
	node_cold = np.loadtxt('stommel_bif_cold.txt',unpack=True)
	node_unstable = np.loadtxt('stommel_bif_unstable.txt',unpack=True)
	spl = UnivariateSpline(node_warm[0], node_warm[1], s=0, k=1)

	etagrid = np.arange(2.65, 3.005, 0.005)
	etagrid1=etagrid;etagrid2=etagrid;etagrid3=etagrid
	warm_intpT = spl(etagrid1)
	spl = UnivariateSpline(node_warm[0], node_warm[2], s=0, k=1)
	warm_intpS = spl(etagrid1)
	spl = UnivariateSpline(node_cold[0], node_cold[1], s=0, k=1)
	cold_intpT = spl(etagrid2)
	spl = UnivariateSpline(node_cold[0], node_cold[2], s=0, k=1)
	cold_intpS = spl(etagrid2)
	spl = UnivariateSpline(node_unstable[0][::-1], node_unstable[1][::-1], s=0, k=1)
	unstable_intpT = spl(etagrid3)
	spl = UnivariateSpline(node_unstable[0][::-1], node_unstable[2][::-1], s=0, k=1)
	unstable_intpS = spl(etagrid3)
	
	unstable_after = unstable_intpT[-1] - unstable_intpS[-1]
	cold_after = cold_intpT[-1] - cold_intpS[-1]
	warm_after = warm_intpT[-1] - warm_intpS[-1]

	h = 0.05; loop = 40.0; T = 5000
	N_T = int(round(float(T)/h/loop))
	t = np.linspace(0, T, N_T)
	eta1 = 2.65; eta2 = 1.; eta3 = 0.3; RT = 1/200.
	params = np.asarray([eta1,eta2,eta3,RT])
	U_0 = [2.4,2.5]
	start = 1000
	ampl = 0.35
	
	dur_vals = [300, 388.5, 389, 500]#Critical value around 388.75
	
	fig=pl.figure()
	for i in range(len(dur_vals)):
		ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur_vals[i]/h)),int(2+(T-start-dur_vals[i])/h)*[ampl]),)
		Te,Se = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
		pl.subplot(len(dur_vals),1,i+1)
		pl.plot(t, len(t)*[unstable_after],'--', color='gold')
		pl.plot(t, len(t)*[cold_after], '-.', color='tomato')
		pl.plot(t, len(t)*[warm_after], ':', color='royalblue')
		pl.plot(t, Te-Se, color='black')
		pl.plot(100*[1000+dur_vals[i]], np.linspace(-0.15, warm_after, 100), color='gray')
		pl.xlabel('Simulation time'); pl.ylabel('q')

def critical_rate():

	h = 0.05; loop = 40.0; T = 15000
	N_T = int(round(float(T)/h/loop))
	t = np.linspace(0, T, N_T)

	eta1 = 2.65; eta2 = 1.; eta3 = 0.3; RT = 1/200.
	params = np.asarray([eta1,eta2,eta3,RT])
	U_0 = [2.4,2.5]

	start = 1000
	ampl_vals = np.linspace(0.30246,0.685, 1000) # bifurcation at 0.683333
	
	d_crit = np.empty(len(ampl_vals))
	for i in range(len(ampl_vals)):
		stop=0
		ampl=ampl_vals[i]
		dur0 = 0.1
		if ampl<0.45:
			dur1 = 1000
		elif ampl<0.55:
			dur1 = 2000
		elif ampl<0.6:
			dur1 = 3000
		elif ampl<0.63:
			dur1 = 5000
		elif ampl<0.63:
			dur1 = 5000
		elif ampl<0.65:
			dur1 = 10000
		elif ampl<0.67:
			dur1 = 20000
		elif ampl<0.675:
			dur1 = 40000
		else:
			dur1 = 100000
		T = dur1 + 5000
		N_T = int(round(float(T)/h/loop))
		
		ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur0/h)),int(2+(T-start-dur0)/h)*[ampl]),)
		Te,Se = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
		if Te[-1]<Se[-1]:
			print('Choose lower dur0')
		ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur1/h)),int(2+(T-start-dur1)/h)*[ampl]),)
		Te,Se = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
		if Te[-1]>Se[-1]:
			print('Choose larger dur1')
			
		while stop==0:
			dur2 = (dur0+dur1)/2.
			ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur2/h)),int(10+(T-start-dur2)/h)*[ampl]),)
			Te,Se = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
			if Te[-1]>Se[-1]:
				dur0=dur2
			else:
				dur1=dur2
			if np.abs(dur0-dur1)<1.:
				print('Result: ', (dur0+dur1)/2.)
				stop=1
			if (dur0+dur1)/2.>99999.:
				d_crit[i] = np.nan
			else:
				d_crit[i] = (dur0+dur1)/2.
        
	fig=pl.figure(figsize=(5.,4.))
	pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
	pl.plot(ampl_vals, d_crit, color='black')
	pl.plot(100*[0.68333], np.linspace(0,100000,100), ':', color='tomato')
	pl.xlabel('$\Delta \eta_1$');pl.ylabel('$D_c$ (years)')

	fig=pl.figure(figsize=(5.,4.))
	pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
	pl.plot(ampl_vals, d_crit, color='black')
	pl.xlabel('$\Delta \eta_1$');pl.ylabel('$D_c$ (years)')
	pl.plot(100*[0.68333], np.linspace(0,100000,100), ':', color='tomato')
	pl.yscale('log')

def linear_decay_rate():
	h = 0.05
	loop = 40.0
	T = 6000 # simulation time in years
	N_T = int(round(float(T)/h/loop))
	t = np.linspace(0, T, N_T)
	
	def f(T,S):
		if T>S:
			return eta1 - T - (T-S)*T
		else:
			return eta1 - T + (T-S)*T
	
	def g(T,S):
		if T>S:
			return eta2 - eta3*S - (T-S)*S
		else:
			return eta2 - eta3*S + (T-S)*S
	
	def eigen(T,S):
		if T>S:
		        tau = -1-eta3-3*T+3*S
		        delt = (-1-2*T+S)*(-eta3-T+2*S)+T*S

		        lamb1 = 0.5*(tau+cmath.sqrt(tau**2-4*delt))
		        lamb2 = 0.5*(tau-cmath.sqrt(tau**2-4*delt))
		        return lamb1.real,lamb1.imag,lamb2.real,lamb2.imag
		else:
		        tau = -1-eta3+3*T-3*S
		        delt = (-1+2*T-S)*(-eta3+T-2*S)+T*S

		        lamb1 = 0.5*(tau+cmath.sqrt(tau**2-4*delt))
		        lamb2 = 0.5*(tau-cmath.sqrt(tau**2-4*delt))
		        return lamb1.real,lamb1.imag,lamb2.real,lamb2.imag

	def eig1_real(T,S):
		return 0.
	
	def jac(T,S):
		if T>S:
			ja = S -2*T -1
			jb = T
			jc = -S
			jd = 2*S-T-eta3

		else:
			ja = 2*T -S - 1
			jb = -T
			jc = S
			jd = T- 2*S -eta3
		return ja, jb, jc, jd
		
			
	eta1 = 2.65
	eta2 = 1.
	eta3 = 0.3
	RT = 1/200. #Inverse timescale of ocean model

	params = np.asarray([eta1,eta2,eta3,RT])

	start = 2000
	dur = 300
	ampl = 0.35
	ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur/h)),int((T-start-dur)/h)*[ampl]),)

	U_0 = [3.,3.55]
	Te,Se = ode_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)

	lambda1_r = [eigen(Te[i],Se[i])[0] for i in range(len(Te))]
	zero_idx = (np.abs(lambda1_r)).argmin()
	lambda1_i = [eigen(Te[i],Se[i])[1] for i in range(len(Te))]
	lambda2_r = [eigen(Te[i],Se[i])[2] for i in range(len(Te))]
	lambda2_i = [eigen(Te[i],Se[i])[3] for i in range(len(Te))]
	
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

	print(Te[zero_idx]-Se[zero_idx])
	print((1+eta3)/3)
	print(t[zero_idx])

	T_values=np.linspace(0.1,4.5,300)
	S_values=np.linspace(0.,4.6,300)
	crit_vals0 = [[eigen(T_values[i],S_values[j])[0] for i in range(300)] for j in range(300)]
	crit_vals1 = [[eigen(T_values[i],S_values[j])[1] for i in range(300)] for j in range(300)]
	crit_vals2 = [[eigen(T_values[i],S_values[j])[2] for i in range(300)] for j in range(300)]
	crit_vals3 = [[eigen(T_values[i],S_values[j])[3] for i in range(300)] for j in range(300)]
	
	ja = [[jac(T_values[i],S_values[j])[0] for i in range(300)] for j in range(300)]
	jb = [[jac(T_values[i],S_values[j])[1] for i in range(300)] for j in range(300)]
	jc = [[jac(T_values[i],S_values[j])[2] for i in range(300)] for j in range(300)]
	jd = [[jac(T_values[i],S_values[j])[3] for i in range(300)] for j in range(300)]

	print('Real eigenvalue 1 at Saddle (eta1=3.0) = ', eigen(2.825,2.763)[0])
	print('Real eigenvalue 2 at Saddle (eta1=3.0) = ', eigen(2.825,2.763)[2])

	print('Jacobian at Saddle (eta1=3.0) = ', jac(2.825,2.763))

	fig=pl.figure()

	pl.suptitle('Stommel ($\eta_3=0.3$): Real Eigenvalue 1 of Jacobian')

	pl.imshow(crit_vals0, interpolation='nearest', cmap='bwr', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(crit_vals0),vmax=np.max(crit_vals0)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	pl.plot(warm_intpT, warm_intpS, color='black')
	pl.plot(cold_intpT, cold_intpS, color='tomato')
	pl.plot(unstable_intpT, unstable_intpS, ':', color='black')
	

	fig2 =pl.figure()
	pl.suptitle('Stommel model ($\eta_3=0.3$): Eigenvalues of Jacobian')
	ax=pl.subplot(221)
	ax.set_title('$\lambda_1$ real')
	pl.imshow(crit_vals0, interpolation='nearest', cmap='RdBu', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(crit_vals0),vmax=np.max(crit_vals0)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")

	pl.plot(T_values,0.5*(2*T_values+1+eta3/2-np.sqrt((1-eta3)*(2*T_values+1)+eta3**2/4)),':',color='black')
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	pl.plot(Te,Se,color='black')

	ax=pl.subplot(222)
	ax.set_title('$\lambda_1$ imaginary')
	pl.imshow(crit_vals1, interpolation='nearest', cmap='RdBu', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(crit_vals1),vmax=np.max(crit_vals1)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.plot(T_values,(1+T_values-eta3+2*np.sqrt(T_values*(1-eta3))),':',color='black')
	pl.plot(T_values,(1+T_values-eta3-2*np.sqrt(T_values*(1-eta3))),':',color='black')
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	pl.plot(Te,Se,color='black')
	ax=pl.subplot(223)
	ax.set_title('$\lambda_2$ real')
	pl.imshow(crit_vals2, interpolation='nearest', cmap='RdBu', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(crit_vals2),vmax=np.max(crit_vals2)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	pl.plot(Te,Se,color='black')
	ax=pl.subplot(224)
	ax.set_title('$\lambda_2$ imaginary')
	pl.imshow(crit_vals3, interpolation='nearest', cmap='RdBu', origin='lower',norm=MidpointNormalize(midpoint=0.001,vmin=np.min(crit_vals3),vmax=np.max(crit_vals3)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.plot(T_values,(1+T_values-eta3+2*np.sqrt(T_values*(1-eta3))),':',color='black')
	pl.plot(T_values,(1+T_values-eta3-2*np.sqrt(T_values*(1-eta3))),':',color='black')
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	pl.plot(Te,Se,color='black')


	fig2 =pl.figure()
	pl.suptitle('Stommel model ($\eta_3=0.3$): Jacobian')
	ax=pl.subplot(221)
	ax.set_title('df/dT')
	pl.imshow(ja, interpolation='nearest', cmap='RdBu', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(ja),vmax=np.max(ja)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	pl.plot(Te,Se,color='black')

	ax=pl.subplot(222)
	ax.set_title('df/dS')
	pl.imshow(jb, interpolation='nearest', cmap='RdBu', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(jb),vmax=np.max(jb)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	pl.plot(Te,Se,color='black')
	
	ax=pl.subplot(223)
	ax.set_title('dg/dT')
	pl.imshow(jc, interpolation='nearest', cmap='RdBu', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(jc),vmax=np.max(jc)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	pl.plot(Te,Se,color='black')
	
	ax=pl.subplot(224)
	ax.set_title('dg/dS')
	pl.imshow(jd, interpolation='nearest', cmap='RdBu', origin='lower',norm=MidpointNormalize(midpoint=0.001,vmin=np.min(jd),vmax=np.max(jd)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	pl.plot(Te,Se,color='black')

	fig1=pl.figure()
	pl.subplot(311)
	pl.plot(t,Te)
	pl.plot(t,Se,'--')
	pl.subplot(312)
	pl.plot(t,lambda1_r)
	pl.plot(t,lambda2_r)
	pl.plot(100*[t[zero_idx]], np.linspace(min(lambda2_r),max(lambda1_r),100),'--',color='gray')
	pl.subplot(313)
	pl.plot(t,lambda1_i)
	pl.plot(t,lambda2_i)
	
	f_265 = [[f(T_values[i],S_values[j]) for i in range(300)] for j in range(300)]
	g_265 = [[g(T_values[i],S_values[j]) for i in range(300)] for j in range(300)]
	
	eta1 = 3.2
	f_3 = [[f(T_values[i],S_values[j]) for i in range(300)] for j in range(300)]
	g_3 = [[g(T_values[i],S_values[j]) for i in range(300)] for j in range(300)]

	fig=pl.figure()

	pl.subplot(221)
	pl.title('Stommel ($\eta_1=2.65$): Flow in T-direction', fontsize=11)

	pl.imshow(f_265, interpolation='nearest', cmap='bwr', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(f_265),vmax=np.max(f_265)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	
	pl.subplot(222)
	pl.title('Stommel ($\eta_1=2.65$): Flow in S-direction', fontsize=11)

	pl.imshow(g_265, interpolation='nearest', cmap='bwr', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(g_265),vmax=np.max(g_265)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	
	pl.subplot(223)
	pl.title('Stommel ($\eta_1=3.2$): Flow in T-direction', fontsize=11)

	pl.imshow(f_3, interpolation='nearest', cmap='bwr', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(f_3),vmax=np.max(f_3)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	
	pl.subplot(224)
	pl.title('Stommel ($\eta_1=3.2$): Flow in S-direction', fontsize=11)

	pl.imshow(g_3, interpolation='nearest', cmap='bwr', origin='lower',norm=MidpointNormalize(midpoint=0,vmin=np.min(g_3),vmax=np.max(g_3)) , extent=(T_values.min(),T_values.max(),S_values.min(),S_values.max()),aspect="auto")
	pl.xlabel('T'); pl.ylabel('S');pl.colorbar();pl.ylim(min(S_values),max(S_values))
	
	Taxis = np.linspace(1.8, 3.3, 400)
	Saxis = np.linspace(1.8, 3.3, 400)
	T_grid, S_grid = np.meshgrid(Taxis, Saxis)
	
	eta1=2.65
	f_grid = np.asarray([[f(T1, S1) for T1 in Taxis] for S1 in Saxis])
	g_grid = np.asarray([[g(T1, S1) for T1 in Taxis] for S1 in Saxis])
	mag = np.linalg.norm(np.dstack([f_grid, g_grid]), axis=-1)
	
	idx_warm = find_idx(eta1, etagrid1)
	idx_cold = find_idx(eta1, etagrid2)
	idx_ustab = find_idx(eta1, etagrid3)
	
	pl.figure()
	pl.subplot(121)
	pl.title('Stommel ($\eta_1=2.65$) flow', fontsize=14)
	pl.plot(Taxis, Saxis, '--', color='black')
	pl.streamplot(T_grid, S_grid, f_grid, g_grid, color=mag, cmap='plasma', density=2, linewidth=0.7)
	pl.plot(cold_intpT[idx_cold], cold_intpS[idx_cold], 'o', color='black', markersize=8)
	pl.plot(unstable_intpT[idx_ustab], unstable_intpS[idx_ustab], '^', color='black', markersize=8)
	
	pl.xlabel('T'); pl.ylabel('S'); pl.colorbar()
	
	eta1=3.2
	f_grid = np.asarray([[f(T1, S1) for T1 in Taxis] for S1 in Saxis])
	g_grid = np.asarray([[g(T1, S1) for T1 in Taxis] for S1 in Saxis])
	mag = np.linalg.norm(np.dstack([f_grid, g_grid]), axis=-1)
	
	idx_warm = find_idx(eta1, etagrid1)
	idx_cold = find_idx(eta1, etagrid2)
	idx_ustab = find_idx(eta1, etagrid3)
	
	pl.subplot(122)
	pl.title('Stommel ($\eta_1=3.2$) flow', fontsize=14)
	pl.plot(Taxis, Saxis, '--', color='black')
	pl.streamplot(T_grid, S_grid, f_grid, g_grid, color=mag, cmap='plasma', density=2, linewidth=0.7)
	pl.plot(cold_intpT[idx_cold], cold_intpS[idx_cold], 'o', color='black', markersize=8)
	pl.plot(unstable_intpT[idx_ustab], unstable_intpS[idx_ustab], '^', color='black', markersize=8)
	pl.xlabel('T'); pl.ylabel('S'); pl.colorbar()


def fixed_points():

        tol = 0.000001
        eta2=1.
        eta3=0.3
        def f(p):
                x,y = p
                f1 = eta1 - x - x*np.abs(x-y)
                f2 = eta2 - eta3*y - y*np.abs(x-y)
                return [f1,f2]

        eta1_values =np.linspace(2.65,3.6,2000)
        iguess = [[2.48,2.55],[2.0,1.8],[1.8,1.25]]
        branch1T=[];branch1S=[];branch2T=[];branch2S=[];branch3T=[];branch3S=[]; indi=0; bif_idx = 999
        print(iguess[0])
        for i in range(len(eta1_values)):
                eta1=eta1_values[i]
                a1=root(fun=f,x0=np.asarray(iguess[0]),tol=tol,jac=False,method='broyden1')
                #print(a1)
                a1T= a1.x[0]; a1S = a1.x[1]
                a2=root(fun=f,x0=np.asarray(iguess[1]),tol=tol,jac=False,method='broyden1')
                a2T= a2.x[0]; a2S = a2.x[1]
                a3=root(fun=f,x0=np.asarray(iguess[2]),tol=tol,jac=False,method='broyden1')
                a3T= a3.x[0]; a3S = a3.x[1]
                branch1T.append(a1T)
                branch1S.append(a1S)
                branch2T.append(a2T)
                branch2S.append(a2S)
                branch3T.append(a3T)
                branch3S.append(a3S)
                iguess=[[a1T,a1S],[a2T,a2S],[a3T,a3S]]
                if (a1T<2.0 and indi==0):
                        bif_idx = i
                        indi = 1
                        print('bifurcation point: ', eta1_values[i])

        fig1=pl.figure()
        pl.subplot(121)
        pl.plot(eta1_values,branch1T)
        pl.plot(eta1_values,branch3T)
        pl.plot(eta1_values,branch2T,'--')
        pl.subplot(122)
        pl.plot(eta1_values,branch1S)
        pl.plot(eta1_values,branch3S)
        pl.plot(eta1_values,branch2S,'--')

def calc_ews_filtered(x1,x2,window,n):
        return [[estimated_ac1_alt2(x1[n*i:n*i+window]) for i in range(int((len(x1)-window)/n))],[estimated_ac1_alt2(x1[n*i:n*i+window]) for i in range(int((len(x2)-window)/n))],[ews.cy_var(x1[n*i:n*i+window]) for i in range(int((len(x1)-window)/n))],[ews.cy_var(x2[n*i:n*i+window]) for i in range(int((len(x2)-window)/n))],[ews.estimated_cc(x1[n*i:n*i+window],x2[n*i:n*i+window]) for i in range(int((len(x1)-window)/n))],[ews.cy_skew(x1[n*i:n*i+window]) for i in range(int((len(x1)-window)/n))],[ews.cy_asym(x1[n*i:n*i+window],1) for i in range(int((len(x1)-window)/n))], [PermEn(x1[n*i:n*i+window]) for i in range(int((len(x1)-window)/n))],[ews.cy_higher_ac(x1[n*i:n*i+window]) for i in range(int((len(x1)-window)/n))]]

def calc_ews_spline(data1,data2,window,n,lag):
        return [[ews.estimated_ac1(detrend_cubic(data1[n*i:n*i+window]),lag) for i in range(int((len(data1)-window)/n))],[ews.estimated_ac1(detrend_cubic(data2[n*i:n*i+window]),lag) for i in range(int((len(data1)-window)/n))],[ews.cy_var(detrend_cubic(data1[n*i:n*i+window])) for i in range(int((len(data1)-window)/n))],[ews.cy_var(detrend_cubic(data2[n*i:n*i+window])) for i in range(int((len(data1)-window)/n))],[ews.estimated_cc(detrend_cubic(data1[n*i:n*i+window]),detrend_cubic(data2[n*i:n*i+window])) for i in range(int((len(data1)-window)/n))]]

def simulate():
        h = 0.05
        loop = 5.
        T = 20000 #simulation time in years
        N_T = int(round(float(T)/h/loop))
        t = np.linspace(0, T, N_T)

        eta1 = 2.65
        eta2 = 1.
        eta3 = 0.3
        RT = 1/200. # Inverse timescale of ocean model

        sigT=0.001; sigS=0.001

        params = np.asarray([eta1,eta2,eta3,RT,sigT,sigS])

        start = 2000
        dur = 300
        ampl = 0.35
        ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur/h)),int((T-start-dur)/h)*[ampl]),)

        U_0 = [1.5,1.0]
        Te,S = sde_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)

        cut = 10000
        np.save('stommel_T_warm_sig001',Te[cut:])
        np.save('stommel_S_warm_sig001',S[cut:])

        pl.plot(t[cut:],Te[cut:])
        pl.plot(t[cut:],S[cut:],'--')
        return(Te,S)

def smoothing_filter_rect(x, N):
        return np.convolve(x, np.ones((N,))/N, mode='same')

def detrend_linear(x, N):
        x_pad  = np.pad(x,int(N/2.),mode='reflect')
        x_detr = np.empty(len(x))
        for i in range(len(x)):
                x_detr[i] = detrend(x_pad[i:i+int(N/2.)])
        return x_detr

def detrend_cubic(y):
        x = np.linspace(0,len(y),len(y))
        model = np.polyfit(x, y, 3)
        trend = np.polyval(model, x)
        return y-trend

def calc_mean_conf(data,length,samples):
        mean = [np.mean([data[j][i] for j in range(samples)]) for i in range(length)]
        percentile_2 = [np.percentile([data[j][i] for j in range(samples)],5.) for i in range(length)]
        percentile_97 = [np.percentile([data[j][i] for j in range(samples)],95.) for i in range(length)]
        return mean, percentile_2, percentile_97

class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self,vmin,vmax,clip)
        def __call__(self, value, clip=None):
                x, y = [self.vmin, self.midpoint, self.vmax], [0,0.5,1]
                return np.ma.masked_array(np.interp(value,x,y),np.isnan(value))

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
	kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
	kde_skl.fit(x[:, np.newaxis])
	log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
	return np.exp(log_pdf)

def calc_stat_kernel_density(data, data_grid, bandwidth):
	kernel_pdf = kde_sklearn(data, data_grid, bandwidth=bandwidth)
	return kernel_pdf
	
def find_idx(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx#array[idx]

def biplot(score,coeff,pcax,pcay,labels=None):
        pca1=pcax-1
        pca2=pcay-1
        xs = score[:,pca1]
        ys = score[:,pca2]
        n=2
        scalex = 1.0/(xs.max()- xs.min())
        scaley = 1.0/(ys.max()- ys.min())
        fig=pl.figure()
        pl.scatter(xs*scalex,ys*scaley)
        for i in range(n):
                pl.arrow(0, 0, coeff[pca1,i], coeff[pca2,i],color='r',alpha=0.5)
                if labels is None:
                        pl.text(coeff[pca1,i]* 1.15, coeff[pca2,i] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
                else:
                        pl.text(coeff[pca1,i]* 1.15, coeff[pca2,i] * 1.15, labels[i], color='g', ha='center', va='center')
        pl.xlim(-1,1)
        pl.ylim(-1,1)
        pl.xlabel("PC{}".format(pcax))
        pl.ylabel("PC{}".format(pcay))
        pl.grid()
        
def reg_m(y, x):
	x = np.array(x).T
	results = sm.OLS(endog=y, exog=x).fit()
	return results
	
def reg_m_new(y, x):
	x = np.array(x).T
	reg = LinearRegression(fit_intercept=False).fit(x, y)
	return reg.coef_
		
def calc_jacobian(deltaT, deltaS, resT, resS):
	predictor = [resT[2:-2],resS[2:-2]]
	results = reg_m(deltaT,predictor,)
	a,b = results.params
	results = reg_m(deltaS,predictor,)
	c,d = results.params
	return a, b, c, d
	
def calc_eigenvalue_all(resT, resS):
	lambdas_r1 = np.empty(len(resT))
	lambdas_r2 = np.empty(len(resT))
	lambdas_i1 = np.empty(len(resT))
	lambdas_i2 = np.empty(len(resT))
	for i in range(len(resT)):
		resT0 = np.asarray(resT[i]); resS0 = np.asarray(resS[i])
		### one-point derivative
		#deltaT0 = resT0[1:]-resT0[:-1]
		#deltaS0 = resS0[1:]-resS0[:-1]
		### two-point derivative
		#deltaT0 = resT0[2:]-resT0[:-2]
		#deltaS0 = resS0[2:]-resS0[:-2]
		### five-point derivative
		deltaT0 = 8.*resT0[3:-1]-8.*resT0[1:-3] + resT0[:-4] - resT0[4:]
		deltaS0 = 8.*resS0[3:-1]-8.*resS0[1:-3] + resS0[:-4] - resS0[4:]
		a, b, c, d = calc_jacobian(deltaT0, deltaS0, resT0, resS0)
		#print(a, b, c, d)
		lamb1 = 0.5*(a+d+cmath.sqrt((a+d)**2-4*(a*d-b*c)))
		lamb2 = 0.5*(a+d-cmath.sqrt((a+d)**2-4*(a*d-b*c)))
		lambdas_r1[i] = lamb1.real
		lambdas_r2[i] = lamb2.real
		lambdas_i1[i] = lamb1.imag
		lambdas_i2[i] = lamb2.imag
	return lambdas_r1, lambdas_r2, lambdas_i1, lambdas_i2
		
	
def calc_ews_running(T, S, thin, window):
	x = np.empty(int(len(T)/thin) - int(window/thin))
	for i in range(int(window/thin), int((len(T))/thin)):
		resT = detrend_cubic(T[thin*i-window:thin*i])
		resS = detrend_cubic(S[thin*i-window:thin*i])
		x[i-int(window/thin)] = 100.*calc_jacobian_neighbors(resT, resS)
	return x
		
def calc_lambda_running(T, S, thin, window):
	def calc_lambda(deltaT, deltaS, resT, resS):
		predictor = [resT[:-1],resS[:-1]]
		results = reg_m(deltaT,predictor,)
		a,b = results.params
		results = reg_m(deltaS,predictor,)
		c,d = results.params

		lamb1 = 0.5*(a+d+cmath.sqrt((a+d)**2-4*(a*d-b*c)))
		return lamb1.real
	
	lambdas = np.empty(int(len(T)/thin) - int(window/thin))
	for i in range(int(window/thin), int((len(T))/thin)):
		resT = detrend_cubic(T[thin*i-window:thin*i])
		resS = detrend_cubic(S[thin*i-window:thin*i])
		deltaT = resT[1:]-resT[:-1]
		deltaS = resS[1:]-resS[:-1]
		lambdas[i-int(window/thin)] = calc_lambda(deltaT, deltaS, resT, resS)
	return lambdas
	
def calc_jacobian_neighbors(xall, yall):
	w = len(xall)
	n = int(w/2.) # consider n nearest data points.
	a = b = c = d = 0.
	for i in range(w-1):
		x0 = xall[i]
		y0 = xall[i]
		### remove x0 and y0 from x and y
		x = np.delete(xall, i); y = np.delete(yall, i)
		distx = x-x0
		disty = y-y0

		dist = distx[:-1]**2 + disty[:-1]**2 # Prevent last value from being chosen

		idcs = np.argpartition(dist, n)[:n]
		distx_t1 = x[idcs+1] - x[idcs]
		disty_t1 = y[idcs+1] - y[idcs]
		
		predictor = [distx[idcs],disty[idcs]]

		a0,b0 = reg_m_new(distx_t1,predictor)
		c0,d0 = reg_m_new(disty_t1,predictor)
		a+=a0; b+=b0; c+=c0; d+=d0
		
	a=a/(w-1); b=b/(w-1); c=c/(w-1); d=d/(w-1)

	return b-c
	
def calc_cov_eigenvalues(x, y):
	cov_matrix = np.cov(np.asarray([x,y]))
	a=cov_matrix[0][0]; b=cov_matrix[0][1]; c=cov_matrix[1][0]; d=cov_matrix[1][1]
	lamb1 = 0.5*(a+d+cmath.sqrt((a+d)**2-4*(a*d-b*c)))
	lamb2 = 0.5*(a+d-cmath.sqrt((a+d)**2-4*(a*d-b*c)))
	return lamb1.real, lamb2.real, lamb1.imag, lamb2.imag

def vector_ar1(x, y):
	data = np.asarray([x,y]).transpose()
	model = VAR(data)
	results = model.fit(1)
	coef_matr = results.coefs[0]
	a=coef_matr[0][0]; b=coef_matr[0][1]; c=coef_matr[1][0]; d=coef_matr[1][1]

	lamb1 = 0.5*(a+d+cmath.sqrt((a+d)**2-4*(a*d-b*c)))
	lamb2 = 0.5*(a+d-cmath.sqrt((a+d)**2-4*(a*d-b*c)))

	return lamb1.real, lamb2.real, lamb1.imag, lamb2.imag
	
def ac_function(slices):
	n = len(slices)
	m = 25 # max lag
	ac_func = np.empty(m)
	for j in range(m):
		ac=0.
		for i in range(n):
			ac += ews.estimated_ac1(slices[i],j)
		ac_func[j] = ac/n
	return ac_func
		
def cc_function(slT, slS):
	n = len(slT)
	m = 25 # max lag
	cc_func = np.empty(2*m+1)
	for j in range(-m,m+1):
		cc=0.
		for i in range(n):
			T = slT[i]; S = slS[i]
			if m==j:
				cc += ews.estimated_cc(T[m+j:], S[m:-m])
			else:
				cc += ews.estimated_cc(T[m+j:-m+j], S[m:-m])
		cc_func[j] = cc/n
	return cc_func
	
def asym_function(slices):
	n = len(slices)
	m = 25 # max lag
	asym_func = np.empty(m)
	for j in range(m):
		asym=0.
		for i in range(n):
			asym += ews.cy_asym(slices[i],j)
		asym_func[j] = asym/n
	return asym_func

def calc_pdf(data):
	xgrid = np.linspace(min(data)-np.std(data)/20.,max(data)+np.std(data)/20.,1000)
	kde = KernelDensity(kernel='gaussian', bandwidth=np.std(data)/20.).fit(np.asarray(data)[:, np.newaxis])
	log_dens = kde.score_samples(xgrid[:, np.newaxis])
	return xgrid, np.exp(log_dens)
	
if __name__ == "__main__":
        MAIN()
