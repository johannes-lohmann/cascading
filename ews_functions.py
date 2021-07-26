import numpy as np
from scipy.ndimage import filters
from scipy.signal import gaussian, detrend, butter, filtfilt
import scipy.signal as ss
import statsmodels.api as sm
from pylab import rand
from scipy.optimize import fmin
import warnings

def detrend_cubic(y):
        x = np.linspace(0,len(y),len(y))
        model = np.polyfit(x, y, 3)
        trend = np.polyval(model, x)
        return y-trend

def detrend_spline(y, order):
        x = np.linspace(0,len(y),len(y))
        model = np.polyfit(x, y, order)
        trend = np.polyval(model, x)
        return y-trend

def smoothing_filter_rect(x, N):
        return np.convolve(x, np.ones((N,))/N, mode='same')

def smoothing_filter_gauss(x,stdev,N):

        wind = gaussian(stdev,N)# stdev and window size
        return filters.convolve(x, wind/wind.sum(), mode='nearest')

def butterworth_zerophase(x, cutoff=0.05):
        b, a = butter(4, cutoff) # order, cut-off (frac. of Nyquist)
        fgust = filtfilt(b, a, x, method="gust")
        return fgust

def sinc_lowpass(data,filt):
        fc = filt
        b = 0.5/100
        N = int(np.ceil((4 / b))) # kernel width
        if not N % 2: N += 1  # Make sure that N is odd.
        n = np.arange(N)
        
        # Compute a low-pass filter.
        h = np.sinc(2 * fc * (n - (N - 1) / 2.)) 
        w = np.blackman(N) 
        h = h * w
        h = h / np.sum(h)

        low = filters.convolve1d(data, h)
        return low

def sinc_highpass(data,filt):
        fc = filt
        b = 0.5/100
        N = int(np.ceil((4 / b)))
        if not N % 2: N += 1  # Make sure that N is odd.
        n = np.arange(N)
        
        # Compute a low-pass filter.
        h = np.sinc(2 * fc * (n - (N - 1) / 2.)) 
        w = np.blackman(N) 
        h = h * w
        h = h / np.sum(h)

        # Create a high-pass filter from the low-pass filter through spectral inversion.
        h = -h
        h[(N - 1) / 2] += 1
        high = filters.convolve1d(data, h)
        return high


def estimated_ac1_alt2(x):
	ar_mod = sm.tsa.AR(x-np.mean(x), missing='none')
	ar_res = ar_mod.fit(maxlag=1, trend='nc')
	return ar_res.params

def ac_fit_exp(lags,ac):
        fp = lambda c, x: np.exp(-x/c)
        e = lambda p, x, y: ((fp(p,x)-y)**2).sum()
        p0 = rand(1)
        p1 = fmin(e, p0, args=(lags,ac),full_output=False, disp=False)
        return p1

import matplotlib.pyplot as plt


# detrended fluctuation analysis
def DFA(x):

        def calc_rms(x, scale):
            # making an array with data divided in windows
            shape = (x.shape[0]//scale, scale)
            X = np.lib.stride_tricks.as_strided(x,shape=shape)
            #print X
            # vector of x-axis points to regression
            scale_ax = np.arange(scale)
            rms = np.zeros(X.shape[0])
            for e, xcut in enumerate(X):
                coeff = np.polyfit(scale_ax, xcut, 1)
                xfit = np.polyval(coeff, scale_ax)
                # detrending and computing RMS of each window
                rms[e] = np.sqrt(np.mean((xcut-xfit)**2))
            return rms

        def dfa(x, scale_lim=[4,8], scale_dens=0.25, show=False):
            # cumulative sum of data with substracted offset
            y = np.cumsum(x - np.mean(x))
            scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
            # scales = number of data points in window segments.
            fluct = np.zeros(len(scales))
            for e, sc in enumerate(scales):
                fluct[e] = np.mean(calc_rms(y, sc))  #RMSD of the individual segment RMSD's for given scale
            # fitting a line to rms data
            coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
            if show:
                fluctfit = 2**np.polyval(coeff,np.log2(scales))
                plt.loglog(scales, fluct, 'bo')
                plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
                plt.title('DFA')
                plt.xlabel(r'$\log_{10}$(time window)')
                plt.ylabel(r'$\log_{10}$<F(t)>')
                plt.legend()
                plt.show()
            return coeff[0]#scales, fluct, coeff[0]
        result = dfa(x)
        return result

# detrended cross-correlation analysis
def DCCA(x1,x2):

        def calc_cov(z1, z2, scale):
            # making an array with data divided in windows
            ### // = floor division, result is whole numbers
            shape = (z1.shape[0]//scale, scale)
            Z1 = np.lib.stride_tricks.as_strided(z1,shape=shape)
            Z2 = np.lib.stride_tricks.as_strided(z2,shape=shape)
            #print Z1
            # vector of x-axis points to regression
            scale_az = np.arange(scale)
            cov = np.zeros(Z1.shape[0])
            for i in range(len(Z1)):
                coeff1 = np.polyfit(scale_az, Z1[i], 1)
                zfit1 = np.polyval(coeff1, scale_az)
                coeff2 = np.polyfit(scale_az, Z2[i], 1)
                zfit2 = np.polyval(coeff2, scale_az)
                # detrending and computing covariance of each window
                cov[i] = np.mean((Z1[i]-zfit1)*(Z2[i]-zfit2))
            return np.mean(cov)

        def dcca(x1, x2, scale_lim=[4,8], scale_dens=0.25, show=False):
            # cumulative sum of data with substracted offset
            y1 = np.cumsum(x1 - np.mean(x1))
            y2 = np.cumsum(x2 - np.mean(x2))
            scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
            # scales = number of data points in window segments.
            fluct = np.zeros(len(scales))
            for e, sc in enumerate(scales):
                fluct[e] = np.sqrt(np.abs(calc_cov(y1, y2, sc)))
            coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
            if show:
                fluctfit = 2**np.polyval(coeff,np.log2(scales))
                print('alpha = ', coeff[0])
                fig=plt.figure(figsize=(5,4))
                ax=plt.subplot(111)
                ax.tick_params(direction='out', length=6, width=1.5)
                plt.subplots_adjust(left=0.2, bottom=0.17, right=0.97, top=0.97, wspace=0.4, hspace=0.3)
                plt.loglog(scales, fluct, 'bo')
                plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
                plt.xlabel(r'$\log_{10}$(time window)')
                plt.ylabel(r'$\log_{10}$<F(t)>')
                plt.legend()
                plt.show()
            return coeff[0]
        result = dcca(x1,x2)
        return result

