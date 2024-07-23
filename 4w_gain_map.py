"""Kinetic inductance TWPA 2d gain map calculator"""

import tables
from numpy import *
from matplotlib.pyplot import *
import scipy.interpolate as si
from ki_twpa import Gain4w

Ap = 0.08 #Pump amplitude normalized to I*
As = 1e-6 #Signal amplitude normalized to I*
fp = linspace(7.4e9, 8.5e9,100) #Pump frequency
fs = linspace(0, 2.*min(fp), 100) #Signal frequency
tol = 1e-6 #Relative solution tolerance
disp_file = '4w_grAl_Ge.h5' #Dispersion file

#Open dispersion file
file = tables.open_file(disp_file, mode='r')
F = array(file.root.F)
k = array(file.root.k)
l = array(file.root.l)[0]
k_func = si.interp1d(F,k)

g4w = Gain4w(l, k_func) 
G = g4w.solve(fs,fp, Ap)
figure('Gain map')
pcolormesh(fs,fp, G)
clim((0,25))
colorbar()
grid()
title("Gain, dB")
xlabel("Fs")
ylabel("Fp")
show()

