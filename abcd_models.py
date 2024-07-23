from numpy import *
from matplotlib.pyplot import *
from numpy.random import default_rng
from numba import njit
import scipy.constants as sc

@njit
def disp_4w_tl(F, z_alter, l_alter, z3_alter, l3_alter, fp, v, length, Z0 = 50.):
	abcd_tl = lambda Z0,g,l: array([ [cosh(g*l), Z0*sinh(g*l)],[sinh(g*l)/Z0, cosh(g*l)] ])
	S21_abcd = lambda A, Z0: 2/(A[0,0] + A[0,1]/Z0 + A[1,0]*Z0 + A[1,1])
	per = v/(3.*fp*2)
	Ncells = int(length/(per*3))
	A_seg = zeros( (6,2,2), dtype = complex128)
	S21 = zeros(len(F), dtype = complex128)
	for j,f in enumerate(F):
		g = 1.j* 2*pi*f / v
		A_seg[0] = abcd_tl( Z0*z_alter, g, l_alter*per )
		A_seg[1] = abcd_tl( Z0, g, per*(1.-l_alter) )
		A_seg[2] = abcd_tl( Z0*z_alter, g, l_alter*per )
		A_seg[3] = abcd_tl( Z0, g, per*(1.-l_alter/2-l3_alter/2) )
		A_seg[4] = abcd_tl( Z0*z3_alter, g, l3_alter*per )
		A_seg[5] = abcd_tl( Z0, g, per*(1.-l_alter/2-l3_alter/2) )
		for i,A in enumerate(A_seg):
			if not i: Acell = A
			else: Acell = dot(Acell, A)
		for i in range(Ncells):
			if not i: Atot = Acell 
			else: Atot = dot(Atot, Acell)
		S21[j] = S21_abcd(Atot, Z0)
	l_final = Ncells*2.*per	
	return S21, l_final, per
	
@njit
def disp_3w_tl(F, z_alter, l_alter, z2_alter, l2_alter, fp, v, length, Z0 = 50.):
	abcd_tl = lambda Z0,g,l: array([ [cosh(g*l), Z0*sinh(g*l)],[sinh(g*l)/Z0, cosh(g*l)] ])
	S21_abcd = lambda A, Z0: 2/(A[0,0] + A[0,1]/Z0 + A[1,0]*Z0 + A[1,1])
	per = v/(2.*fp*2)
	Ncells = int(length/(per*2))
	print(Ncells)
	A_seg = zeros( (4,2,2), dtype = complex128)
	S21 = zeros(len(F), dtype = complex128)
	for j,f in enumerate(F):
		g = 1.j* 2*pi*f / v
		A_seg[0] = abcd_tl( Z0*z_alter, g, l_alter*per )
		A_seg[1] = abcd_tl( Z0, g, per*(1.-l_alter/2-l2_alter/2) )
		A_seg[2] = abcd_tl( Z0*z2_alter, g, l2_alter*per )
		A_seg[3] = abcd_tl( Z0, g, per*(1.-l_alter/2-l2_alter/2) )
		for i,A in enumerate(A_seg):
			if not i: Acell = A
			else: Acell = dot(Acell, A)
		for i in range(Ncells):
			if not i: Atot = Acell 
			else: Atot = dot(Atot, Acell)
		S21[j] = S21_abcd(Atot, Z0) 
	#print( (Z0*z_alter*l_alter + Z0*(2.-l_alter-l2_alter) + Z0*z2_alter*l2_alter)/2 )
	l_final = Ncells*2.*per	
	return S21, l_final, per
	
class ModulatedTl():
	def __init__(self):
		self.v = 0. 		#Propagation velocity
		self.Z0 = 50.		#Line impedance
		self.l = 50e-3		#Line length
		self.Z_mod = 1. 	#Impedance modulation, Z = Z_mod*Z0
		self.l_mod = 0.1 	#Length of Zmod segment, l = l_mod*per 
		self.Z1_mod = 1. 	#Impedance modulation of every 2th or 3th segment 
		self.l1_mod = 0.1 	#Length of every 2th or 3th segment
		
	def delta_k_4w(self, fp, Ap):
		return Ap**2/8.
	
	def f_points_gen(self, fp, spans, step_deviders):
		f_step = self.v/(3.*self.l)
		F = array([])
		for i,span in enumerate(spans):
			f = fp*(i+1)
			if i:
				F = hstack( (F, arange(fp*i+spans[i-1]/2, f-span/2., f_step)) )
			else:
				F = hstack( (F, arange(0., f-span/2., f_step)) )
			F = hstack( (F, arange(f-span/2., f+span/2., f_step/step_deviders[i])) )
		return F
	
	def _calc_delta_k(self, F, S21, l_final):
		delta_k = -unwrap(angle(S21))/(F*2.*pi*l_final/self.v) - 1
		return delta_k - mean(delta_k[1:100])
		
	def disp_3w(self,F, fp):
		Z = self.Z0*sqrt( (self.l_mod/self.Z_mod + 2.*(1.-self.l1_mod/2.-self.l_mod/2.) + self.l1_mod/self.Z1_mod )/
							(self.l_mod*self.Z_mod + 2.*(1.-self.l1_mod/2.-self.l_mod/2.) + self.l1_mod*self.Z1_mod ))					
		S21, l_final, per = disp_3w_tl(F, self.Z_mod, self.l_mod, self.Z1_mod, self.l1_mod, fp, self.v, self.l, Z)
		delta_k = self._calc_delta_k(F, S21, l_final)
		return S21, delta_k, per, Z
		
	def disp_4w(self,F, fp):
		Z = self.Z0*sqrt( (2.*self.l_mod/self.Z_mod + 2*(1.-self.l1_mod/2.-self.l_mod/2.) + (1.-self.l_mod) + self.l1_mod/self.Z1_mod )/
							(2.*self.l_mod*self.Z_mod  + 2.*(1.-self.l1_mod/2.-self.l_mod/2.) + (1.-self.l_mod) + self.l1_mod*self.Z1_mod ))					
		S21, l_final, per = disp_4w_tl(F, self.Z_mod, self.l_mod, self.Z1_mod, self.l1_mod, fp, self.v, self.l, Z)
		delta_k = self._calc_delta_k(F, S21, l_final)
		return S21, delta_k, per, Z, l_final	
	