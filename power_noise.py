# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from numpy.fft import fft
from numpy.fft import ifft
from numpy import conjugate

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from astroML.time_series import generate_power_law
from astroML.fourier import PSD_continuous

import astropy
from astropy.io import fits
from astropy.io import ascii

from random import seed
from random import gauss

import math
import csv
#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)


# Open data
FITS = fits.open('./srcB_3to40_cl_barycorr_binned_multiD.fits')

data = astropy.table.Table.read(FITS[1])
data = FITS[1].data
DATA_SIZE = len(data)
print("LENGTH: ", DATA_SIZE)

t_obs = data.field(0)
col = data.field(2)
t_obs = t_obs[:-1]

x_obs = col[:,0]
x_obs = x_obs[:-1]


# Make Power Spectrum
N = 1024  
dt = 0.01
factor = 100

t = dt * np.arange(N)
random_state = np.random.RandomState(1)

for i, beta in enumerate([1.0, 2.0]):
    # Generate the light curve and compute the PSD
    x = factor * generate_power_law(N, dt, beta, random_state=random_state)
    f, PSD = PSD_continuous(t, x)

    print("With Beta of ", beta)
    print("F looks like: ", f[:20])
    print("PSD looks like: ", PSD[:20])


# Timmer and Konig Algorithm:
seed(42)
generated_f_pts = []

for w_i in range(len(f)):
    n1 = gauss(0, 1)
    n2 = gauss(0, 1)

    real_part = n1 * math.sqrt(0.5 * f[w_i])
    imaginary_part = n2 * math.sqrt(0.5 * f[w_i])

    generated_f_pts.append(complex(real_part, imaginary_part))
  


FFT = fft(x_obs)  # take FFT of original data

final_f_signal = []

for i, f_pt in enumerate(FFT):
    if x_obs[i] < 150:
        final_f_signal.append(generated_f_pts[i])
    else:
        final_f_signal.append(FFT[i])


final_f_signal = [x if x.imag >=0 else conjugate(x) for x in final_f_signal]  # fix negative iamginary parts
print("final signal looks like: ", final_f_signal[:20])

sig = final_f_signal#ifft(final_f_signal)  # reverse the FFT to get back time series data
print("Got back a signal of size: ", len(sig))



# Output data to csv file
l = [('t_obs', 'x_obs', 'final signal'), (t_obs, x_obs, sig)]

with open('psd1.csv', 'w', ) as myfile:
    wr = csv.writer(myfile)
#    zipped_list = zip(*l)
    wr.writerow(l[0])
    for i in range(len(sig)):
        wr.writerow([t_obs[i], x_obs[i], math.sqrt((sig[i].real)**2 + (sig[i].imag)**2)])
print("Wrote data to file")


