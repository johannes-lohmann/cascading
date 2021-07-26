import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import root
import stommel_cy_noise as sde_cy
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline

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

eta3=0.2
eta1=3.0

def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx

def bif_diagram_numeric():

        eta2 = 1.; eta3 = 0.3
        eta1 = 3.
        def fa(x):
                T = x[0]; S = x[1]
                return[-T+eta1 -np.abs(T-S)*T, -eta3*S + eta2 -np.abs(T-S)*S]

        #eta1_values =np.linspace(2.6,3.3,2000)
        eta2_values =np.linspace(0.91,1.5,2000)
        #iguess1 = [2.35,2.5]
        #iguess2 = [2.1,2.0]
        #iguess3 = [0.5,0.4]
        iguess1 = [3.2,3.2]
        iguess2 = [2.96,2.95]
        iguess3 = [1.63,0.8]
        branch1T=[];branch2T=[];branch3T=[];branch1S=[];branch2S=[];branch3S=[]; indi=0; bif_idx = len(eta2_values)
        for i in range(len(eta2_values)):
                #eta1=eta1_values[i]
                eta2=eta2_values[i]
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

        #np.save('stommel_bif_fine',np.asarray([eta1_values,branch1T,branch1S,branch2T,branch2S,branch3T,branch3S]))

        #fig2=pl.figure()
        #pl.subplot(121)
        #pl.plot(eta2_values,branch1T)
        #pl.plot(eta2_values[:bif_idx],branch3T[:bif_idx])
        #pl.plot(eta2_values[:bif_idx],branch2T[:bif_idx],':')
        #pl.subplot(122)
        pl.plot(eta2_values,branch1S)
        pl.plot(eta2_values[:bif_idx],branch3S[:bif_idx])
        pl.plot(eta2_values[:bif_idx],branch2S[:bif_idx],':')

def simulate():
        h = 0.05; loop = 5.; T = 10000#5500#87 #simulation time in years
        N_T = int(round(float(T)/h/loop)); t = np.linspace(0, T, N_T)

        eta1 = 3.; eta2 = 1.; eta3 = 0.2
        RT = 1/200. #inverse timescale of ocean model
        sigT=0.; sigS=0.#0.001
        params = np.asarray([eta1,eta2,eta3,RT,sigT,sigS])

        start = 2000
        dur = 700
        ampl = -0.3
        ramp = np.concatenate((int(start/h)*[0.],np.linspace(0.,ampl,int(dur/h)),int((T-start-dur)/h)*[ampl]),)

        U_0 = [2.7,2.8]
        Te,S = sde_cy.solver(np.asarray(U_0), N_T, int(loop), h, params, ramp)
        print(len(Te))

        cut = int(start/h/loop)#10000
        #np.save('stommel_T_warm_sig001',Te[cut:])
        #np.save('stommel_S_warm_sig001',S[cut:])

        #pl.plot(t[cut:],Te[cut:])
        #pl.plot(t[cut:],S[cut:],'--')
        #pl.plot(t,ramp[::int(loop)])
        ramp0=ramp[::int(loop)]
        return t[cut:], Te[cut:], S[cut:], ramp0[cut:]

#def plot_bif_diagram_S(eta3):
#	pl.plot([eta2_s_1(x) for x in np.linspace(.1,1.591,1000)], np.linspace(.1,1.591,1000), color='black')
#	pl.plot([eta2_s_1(x) for x in np.linspace(1.591,3.,1000)], np.linspace(1.591,3.,1000),':', color='black')
#	pl.plot([eta2_s_2(x) for x in np.linspace(.1,3.,1000)], np.linspace(.1,3.,1000), color='tomato')
#	pl.plot([eta2_s_3(x) for x in np.linspace(.1,3.,1000)], np.linspace(.1,3.,1000), color='forestgreen')
#	pl.plot([eta2_s_4(x) for x in np.linspace(.1,3.,1000)], np.linspace(.1,3.,1000), color='royalblue')
	
#def plot_bif_diagram_T(eta3):
#	pl.plot([eta2_t_1(x) for x in np.linspace(.5,2.052,1000)], np.linspace(.5,2.052,1000), color='black')
#	pl.plot([eta2_t_1(x) for x in np.linspace(2.052,3.,1000)], np.linspace(2.052,3.,1000),':', color='black')
#	pl.plot([eta2_t_2(x) for x in np.linspace(.5,3.0,1000)], np.linspace(.5,3.,1000), color='tomato')
	
	
def eta2_s_1(s):
	return -0.5*s**2+(eta3-0.5)*s + s*np.sqrt(0.25*(s-1)**2+eta1)
		
def eta2_s_2(s):
	return -0.5*s**2+(eta3-0.5)*s - s*np.sqrt(0.25*(s-1)**2+eta1)
	
def eta2_s_3(s):
	if (s+1.)**2<4*eta1:
		return np.nan
	else:
		return 0.5*s**2+(eta3-0.5)*s - s*np.sqrt(0.25*(s+1)**2-eta1)
def eta2_s_4(s):
	if (s+1.)**2<4*eta1:
		return np.nan
	else:
		return 0.5*s**2+(eta3-0.5)*s + s*np.sqrt(0.25*(s+1)**2-eta1)
def eta2_t_1(t):
	return eta3 + eta1 - 1 + (eta3-1)*t + (2*eta1-eta1*eta3)/t - (eta1/t)**2
def eta2_t_2(t):
	return eta1 - eta3 + 1 + (eta3-1)*t - (2*eta1-eta1*eta3)/t + (eta1/t)**2


def eta2_q_1(bpoint):
	S_vals = np.linspace(.1,bpoint,1000)
	eta2_vals = [eta2_s_1(x) for x in S_vals]
	T_vals=np.empty(len(S_vals))
	for i in range(len(eta2_vals)):
		def fa(T):
			eta2 = eta2_vals[i]
			return (eta3-1)*T**3 + (eta1-eta2+eta3-1)*T**2 + (2*eta1-eta1*eta3)*T -eta1**2
		sol1=root(fa,1.5,jac=False)
		T_vals[i]=sol1.x
	return [eta2_s_1(x) for x in S_vals], T_vals-S_vals
	
def eta2_q_2(bpoint):
	S_vals = np.linspace(bpoint,3.,1000)#1.591
	eta2_vals = [eta2_s_1(x) for x in S_vals]
	T_vals=np.empty(len(S_vals))
	for i in range(len(eta2_vals)):
		def fa(T):
			eta2 = eta2_vals[i]
			return (eta3-1)*T**3 + (eta1-eta2+eta3-1)*T**2 + (2*eta1-eta1*eta3)*T -eta1**2
		sol1=root(fa,2.7,jac=False)
		T_vals[i]=sol1.x
	return [eta2_s_1(x) for x in S_vals], T_vals-S_vals
	
def eta2_q_3():
	S_vals = np.linspace(.1,3.,1000)
	eta2_vals = [eta2_s_3(x) for x in S_vals]
	T_vals=np.empty(len(S_vals))
	for i in range(len(eta2_vals)):
		def fa(T):
			eta2 = eta2_vals[i]
			return (eta3-1)*T**3 + (eta1-eta2-eta3+1)*T**2 - (2*eta1-eta1*eta3)*T + eta1**2
		sol1=root(fa,3.,jac=False)
		T_vals[i]=sol1.x
	return [eta2_s_3(x) for x in S_vals], T_vals-S_vals
	
	

#t, T, S, ramp = simulate()
#eta2_vals1, q_vals1 = eta2_q_1()
#eta2_vals2, q_vals2 = eta2_q_2()
#eta2_vals3, q_vals3 = eta2_q_3()

def plot_q_bif_diagram(bpoint, alph):
	eta2_vals1, q_vals1 = eta2_q_1(bpoint)
	eta2_vals2, q_vals2 = eta2_q_2(bpoint)
	eta2_vals3, q_vals3 = eta2_q_3()
	pl.plot(eta2_vals1, q_vals1, color='black', alpha=alph+0.2)
	pl.plot(eta2_vals2, q_vals2, ':', color='black', alpha=alph+0.2)
	pl.plot(eta2_vals3, q_vals3, color='tomato', alpha=alph+0.2)
fig=pl.figure(figsize=(5.,4.))
pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
eta3=0.75
plot_q_bif_diagram(2.98, eta3)
eta3=0.625
plot_q_bif_diagram(2.53, eta3)
eta3=0.4
plot_q_bif_diagram(1.96, eta3)
eta3=0.2
plot_q_bif_diagram(1.591, eta3)
eta3=0.1
plot_q_bif_diagram(1.44, eta3)
pl.xlabel('$\eta_2$');pl.ylabel('q')
pl.xlim(0.2,2.5);pl.ylim(-0.5,1.25)

fig=pl.figure(figsize=(5.,4.))
pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
eta3=0.3
plot_q_bif_diagram(1.76, .8)
pl.xlabel('$\eta_2$');pl.ylabel('q')
pl.xlim(0.75,1.35);pl.ylim(-0.2,1.)


#fig=pl.figure(figsize=(10.,3.))
#pl.subplots_adjust(left=0.075, bottom=0.18, right=0.98, top=0.97, wspace=0.25, hspace=0.3)
#pl.subplot(131)
#pl.plot(eta2_vals1, q_vals1, color='black')
#pl.plot(eta2_vals2[:-27], q_vals2[:-27], ':', color='black')
#pl.plot(eta2_vals3, q_vals3, color='tomato')
#pl.plot(1.+ramp, T-S, color='forestgreen', linewidth=2.5)
#pl.xlim(0.2,1.5);pl.ylim(-0.5,1.25)
#pl.xlabel('$\eta_2$');pl.ylabel('q')

fig=pl.figure(figsize=(5.,4.))
pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
#plot_bif_diagram_T(0.2)
pl.plot([eta2_t_1(x) for x in np.linspace(1.5,2.155,1000)], np.linspace(1.5,2.155,1000), color='black')
pl.plot([eta2_t_1(x) for x in np.linspace(2.155,3.,1000)], np.linspace(2.155,3.,1000),':', color='black')
pl.plot([eta2_t_2(x) for x in np.linspace(1.5,3.0,1000)], np.linspace(1.5,3.,1000), color='tomato')
#pl.plot(1.+ramp, T, color='forestgreen', linewidth=2.5)
pl.xlim(0.75,1.35)
pl.xlabel('$\eta_2$');pl.ylabel('T')

fig=pl.figure(figsize=(5.,4.))
pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
#plot_bif_diagram_S(0.2)
pl.plot([eta2_s_1(x) for x in np.linspace(.5,1.763,1000)], np.linspace(.5,1.763,1000), color='black')
pl.plot([eta2_s_1(x) for x in np.linspace(1.763,3.,1000)], np.linspace(1.763,3.,1000),':', color='black')
pl.plot([eta2_s_3(x) for x in np.linspace(.5,3.,1000)], np.linspace(.5,3.,1000), color='tomato')

#pl.plot(1.+ramp, S, color='forestgreen', linewidth=2.5)
pl.xlabel('$\eta_2$');pl.ylabel('S')
pl.xlim(0.75,1.35)

'''
fig=pl.figure()

eta3=0.1
pl.plot([eta2_t_1(x) for x in np.linspace(.5,1.95,1000)], np.linspace(.5,1.95,1000), color='black', alpha=0.3)
pl.plot([eta2_t_1(x) for x in np.linspace(1.95,3.,1000)], np.linspace(1.95,3.,1000),':', color='black', alpha=0.3)
pl.plot([eta2_t_2(x) for x in np.linspace(.5,3.0,1000)], np.linspace(.5,3.,1000), color='tomato', alpha=0.3)

eta3=0.2
pl.plot([eta2_t_1(x) for x in np.linspace(.5,2.052,1000)], np.linspace(.5,2.052,1000), color='black', alpha=0.4)
pl.plot([eta2_t_1(x) for x in np.linspace(2.052,3.,1000)], np.linspace(2.052,3.,1000),':', color='black', alpha=0.4)
pl.plot([eta2_t_2(x) for x in np.linspace(.5,3.0,1000)], np.linspace(.5,3.,1000), color='tomato', alpha=0.4)

eta3=0.4
pl.plot([eta2_t_1(x) for x in np.linspace(.5,2.4,1000)], np.linspace(.5,2.4,1000), color='black', alpha=0.6)
pl.plot([eta2_t_1(x) for x in np.linspace(2.4,3.,1000)], np.linspace(2.4,3.,1000),':', color='black', alpha=0.6)
pl.plot([eta2_t_2(x) for x in np.linspace(.5,3.0,1000)], np.linspace(.5,3.,1000), color='tomato', alpha=0.6)

eta3=0.65
pl.plot([eta2_t_1(x) for x in np.linspace(.5,2.74,1000)], np.linspace(.5,2.74,1000), color='black', alpha=0.8)
pl.plot([eta2_t_1(x) for x in np.linspace(2.74,3.,1000)], np.linspace(2.74,3.,1000),':', color='black', alpha=0.8)
pl.plot([eta2_t_2(x) for x in np.linspace(.5,3.0,1000)], np.linspace(.5,3.,1000), color='tomato', alpha=0.8)

eta3=0.75
pl.plot([eta2_t_1(x) for x in np.linspace(.5,3.,1000)], np.linspace(.5,3.,1000), color='black')
#pl.plot([eta2_t_1(x) for x in np.linspace(2.052,3.,1000)], np.linspace(2.052,3.,1000),':', color='black')
pl.plot([eta2_t_2(x) for x in np.linspace(.5,3.0,1000)], np.linspace(.5,3.,1000), color='tomato')

pl.xlim(0.2,3.5);pl.ylim(1.2,3.2)
pl.xlabel('$\eta_2$');pl.ylabel('T')
'''

#bif_diagram_numeric()







eta3=0.3; eta2=1.0

def eta1_t_1(t):
	return -0.5*t**2+(1-eta3*0.5)*t + t*np.sqrt(0.25*(t-eta3)**2+eta2)
	
def eta1_t_2(t):
	return -0.5*t**2+(1-eta3*0.5)*t - t*np.sqrt(0.25*(t-eta3)**2+eta2)
\
def eta1_t_3(t):
	if (t+eta3)**2<4.*eta2:
		return np.nan
	else:
		return 0.5*t**2+(1-0.5*eta3)*t - t*np.sqrt(0.25*(t+eta3)**2-eta2)
		
def eta1_t_4(t):
	if (t+eta3)**2<4.*eta2:
		return np.nan
	else:
		return 0.5*t**2+(1-0.5*eta3)*t + t*np.sqrt(0.25*(t+eta3)**2-eta2)
		
def eta1_s_1(s):
	return (1-eta3)*s + eta2 - eta3*(1-eta3) + (eta2-2*eta2*eta3)/s + (eta2/s)**2
	
def eta1_s_2(s):
	return (1-eta3)*s + eta2 + eta3*(1-eta3) - (eta2-2*eta2*eta3)/s - (eta2/s)**2
	


def eta1_q_1(bpoint):
	S_vals = np.linspace(1.6,3.3333,1000)
	eta1_vals = [eta1_s_2(x) for x in S_vals]
	#print(eta1_vals)
	T_vals=np.empty(len(S_vals))
	for i in range(len(eta1_vals)):
		def fa(t):
			eta1 = eta1_vals[i]
			return -0.5*t**2+(1-eta3*0.5)*t + t*np.sqrt(0.25*(t-eta3)**2+eta2) - eta1
		sol1=root(fa,2.0,jac=False)#1.8
		T_vals[i]=sol1.x
	return eta1_vals, T_vals-S_vals
	
def eta1_q_2(bpoint):
	S_vals = np.linspace(0.9,1.554,1000)
	eta1_vals = [eta1_s_1(x) for x in S_vals]
	#print(eta1_vals)
	T_vals0 = np.linspace(1.71,1.9,5000)
	eta1_valsT = [eta1_t_3(x) for x in T_vals0]
	T_vals=np.empty(len(S_vals))
	for i in range(len(eta1_vals)):
		eta1 = eta1_vals[i]
		#def fa(t):
		#	eta1 = eta1_vals[i]
		#	if (t+eta3)**2<4.*eta2:
		#		return np.nan
		#	else:
		#		return 0.5*t**2+(1-0.5*eta3)*t + t*np.sqrt(0.25*(t+eta3)**2-eta2) -eta1
		#sol1=root(fa,1.5,jac=False)#1.8
		#T_vals[i]=sol1.x
		idx = find_nearest(np.asarray(eta1_valsT), eta1)
		#print(T_vals0[idx])
		T_vals[i] = T_vals0[idx]
	return eta1_vals, T_vals-S_vals
	
def eta1_q_3(bpoint):
	S_vals = np.linspace(1.554,3.3333, 1000)
	eta1_vals = [eta1_s_1(x) for x in S_vals]
	#print(eta1_vals)
	T_vals=np.empty(len(S_vals))
	for i in range(len(eta1_vals)):
		def fa(t):
			eta1 = eta1_vals[i]
			if (t+eta3)**2<4.*eta2:
				return np.nan
			else:
				return 0.5*t**2+(1-0.5*eta3)*t - t*np.sqrt(0.25*(t+eta3)**2-eta2) -eta1
		sol1=root(fa,2.0,jac=False)#1.8
		T_vals[i]=sol1.x
	return eta1_vals, T_vals-S_vals
	
	
def eta1_q_4(bpoint):
	S_vals = np.linspace(0.7,0.9,1000)
	eta1_vals = [eta1_s_1(x) for x in S_vals]
	#print(eta1_vals)
	T_vals0 = np.linspace(1.7,1.85,5000)
	eta1_valsT = [eta1_t_4(x) for x in T_vals0]
	#print(eta1_valsT)
	T_vals=np.empty(len(S_vals))
	for i in range(len(eta1_vals)):
		eta1=eta1_vals[i]
		#def fa(t):
		#	eta1 = eta1_vals[i]
		#	return 0.5*t**2+(1-0.5*eta3)*t - t*np.sqrt(0.25*(t+eta3)**2-eta2) -eta1
		#sol1=root(fa,1.8,jac=False)#1.8
		#print(sol1)
		#T_vals[i]=sol1.x
		idx = find_nearest(np.asarray(eta1_valsT), eta1)
		#print(T_vals0[idx])
		T_vals[i] = T_vals0[idx]#[0]
	return eta1_vals, T_vals-S_vals

#eta1_vals1, q_vals1 = eta1_q_1(1.9)
#eta1_vals2, q_vals2 = eta1_q_2(1.9)
#eta1_vals3, q_vals3 = eta1_q_3(1.9)
#eta1_vals4, q_vals4 = eta1_q_4(1.9)

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

fig=pl.figure(figsize=(5.,4.))
pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
#pl.plot(eta1_vals1, q_vals1, color='tomato')
#pl.plot(eta1_vals2, q_vals2, color='black')
#pl.plot(eta1_vals3, q_vals3, ':', color='black')
#pl.plot(eta1_vals4, q_vals4, color='green')

pl.plot(etagrid1, warm_intpT-warm_intpS, color='black')
pl.plot(etagrid2, cold_intpT-cold_intpS, color='tomato')
pl.plot(etagrid3, unstable_intpT-unstable_intpS, ':', color='black')
pl.xlabel('$\eta_1$');pl.ylabel('q')
pl.xlim(2., 3.75)

fig=pl.figure(figsize=(5.,4.))
pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
pl.plot([eta1_t_1(x) for x in np.linspace(1.6,3.3333,1000)], np.linspace(1.6,3.3333,1000), color='tomato')
#pl.plot([eta1_t_2(x) for x in np.linspace(0.1,3.5,1000)], np.linspace(0.1,3.5,1000))
pl.plot([eta1_t_3(x) for x in np.linspace(1.6,3.3333,1000)], np.linspace(1.6,3.3333,1000), ':', color='black')
pl.plot([eta1_t_3(x) for x in np.linspace(1.6,1.9,1000)], np.linspace(1.6,1.9,1000), color='black')
pl.plot([eta1_t_4(x) for x in np.linspace(1.65,1.85,1000)], np.linspace(1.65,1.85,1000), color='black')
pl.xlabel('$\eta_1$');pl.ylabel('T'); pl.xlim(2., 3.75)

fig=pl.figure(figsize=(5.,4.))
pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
pl.plot([eta1_s_1(x) for x in np.linspace(0.7,1.554,1000)], np.linspace(0.7,1.554,1000), color='black')
pl.plot([eta1_s_1(x) for x in np.linspace(1.554,3.3333,1000)], np.linspace(1.554,3.3333,1000), ':',  color='black')
pl.plot([eta1_s_2(x) for x in np.linspace(1.6,3.3333,1000)], np.linspace(1.6,3.3333,1000), color='tomato')
pl.xlabel('$\eta_1$');pl.ylabel('S'); pl.xlim(2., 3.75)


fig=pl.figure(figsize=(5.,4.))
pl.subplots_adjust(left=0.19, bottom=0.16, right=0.96, top=0.97, wspace=0.25, hspace=0.3)
pl.plot(np.linspace(0.7,3.7,100), np.linspace(0.7,3.7,100), '--', color='royalblue')
pl.plot(warm_intpT, warm_intpS, color='black')
pl.plot(cold_intpT, cold_intpS, color='tomato')
pl.plot(unstable_intpT, unstable_intpS, ':', color='black')
pl.xlabel('T'); pl.ylabel('S')

pl.show()
