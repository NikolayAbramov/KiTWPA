"""Dispersion engineering for 4-wave kiTWPA"""

from numpy import *
from matplotlib.pyplot import *
from numpy.random import default_rng
import scipy.constants as sc
from ki_twpa import *
from abcd_models import * 
import tables

project_name = "4w_grAl_Ge"

db = lambda x: 20*log10(x)

ms = KineticMicrostrip()

#Superconducting transition temperature of the material
ms.Tc = 2.15
#Normal resiatance in Ohms per square
ms.Rsq = 500
#Isolator thickness
ms.t = 65e-9
#Isolator dielectric constant
ms.eps = 12.2

#Target pump frequency
fp = 8e9
#F = linspace(0e9, fp-2e9, 3000)
#F = hstack( (F,linspace(fp-2e9, fp+2e9, 4000),linspace(fp+2e9, 3*fp-3e9, 3000), linspace(3*fp-3e9, 3*fp+3e9, 4000) ))

mtl = ModulatedTl()
mtl.v = ms.velocity() 	#Propagation velocity
mtl.l = 70e-3		    #Line length
mtl.Z_mod = 0.8 	    #Impedance modulation, Z = Z_mod*Z0 with period per = v/(3.*fp*2) for harmonics supression
mtl.l_mod = 0.1852 	    #Length of Zmod segment, l = l_mod*per 
mtl.Z1_mod = 0.85 	    #Impedance modulation of every 3th segment for phase matching  
mtl.l1_mod = 0.1852     #Length of every 3th segment

F = mtl.f_points_gen(fp, [2e9, 2e9, 7e9], [4, 4, 4])

#Maximal expected pump amplitude
Ap_max = 0.1
delta_k_req = mtl.delta_k_4w(fp, Ap_max)

S21,delta_k, per,Z, l_final = mtl.disp_4w(F, fp)

file = open(project_name+"_output.txt" , 'w')

file.write('''4-wave TWPA calculation output

Tc =		{:.2f} K
Rsq =		{:.2f} Ohm/sq
t =			{:e} m 
eps =		{:.2f}
v = 		{:e} m/s'''.format(ms.Tc, ms.Rsq, ms.t, ms.eps, ms.velocity()) )

file.write('''
Z0 =		{:.2f} Ohm	
l = 		{:e} m	
Z_mod = 	{:.3f}	
l_mod = 	{:.3f}
Z1_mod = 	{:.3f}
l1_mod = 	{:.3f}'''.format(Z, mtl.l, mtl.Z_mod, mtl.l_mod, mtl.Z1_mod, mtl.l1_mod))

file.write('''
w(Z0) =		{:e} m
period =	{:e} m
'''.format(ms.width(Z), per))

file.write('''
fp =		{:e} Hz
Ap_max =	{:f}
d_kp/kp =	{:f}
'''.format(fp, Ap_max, delta_k_req))

file.close()

file = tables.open_file(project_name+'.h5', mode='w')
file.create_array(file.root, 'S21', S21, "Complex S21")
file.create_array(file.root, 'F', F, "Frequency")
file.create_array(file.root, 'k', -unwrap(angle(S21))/l_final, "k")
file.create_array(file.root, 'l', mtl.l)
file.close()

font = { 'family' : 'Sans',
        'weight' : 'normal',
         'size' : 14}
matplotlib.rc('font', **font)
fig,ax1 = subplots()
ax1.plot(1e-9*F, db(abs(S21)), label = '$S_{21}$')
ax1.set_ylim([-200,0])
ax2 = ax1.twinx()
ax2.plot(1e-9*F, delta_k, color = 'red', label = "${\Delta}k/k$")
ax2.axline( (1e-9*F[0],delta_k_req) , (1e-9*F[-1], delta_k_req), color = 'orange')
ax1.set_ylabel('$S_{21}, dB$')
ax2.set_ylabel("$\Delta k /k$")
ax1.set_xlabel('Frequency, GHz')
ax2.grid()
legend()
tight_layout()
savefig(project_name+".png",dpi=300)
show()