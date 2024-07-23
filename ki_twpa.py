from numpy import *
import scipy.constants as sc
import scipy.integrate as si
from numba import njit
    
@njit
def _coupled_eq_4w(kp, ks, ki, dk, z, A):
    """TWPA coupled wave equations for 4-wave mode according to:
    J. C. Longden and B-K Tan 2024 Eng. Res. Express 6 015068.
    
    Arguments:
    kp -- the pump wavenumber
    ks -- the signal wavenumber
    ki -- the idler wavenumber
    dk -- the wavenumbers relation for 4-w mode: dk = ks + ki - 2*kp
    z -- the coordinate along the amplifier in meters
    A -- the vector of harmonics amplitudes at the coordinate z: A[0] - pump, A[1] - signal, A[2] - idler
        The amplitudes are expressed in relative current units defined as I/I*. 
    """
    Ap = A[0]
    As = A[1]
    Ai = A[2]
    
    dAp_dz = -1.j*kp/8*(Ap*( abs(Ap)**2 + 2*abs(As)**2 + 2*abs(Ai)**2) +
                                2*Ai*As*conj(Ap)*exp(-1.j*dk*z) )
    dAs_dz = -1.j*ks/8*(As*( abs(As)**2 + 2*abs(Ai)**2 + 2*abs(Ap)**2) +
                                conj(Ai)*Ap**2*exp(1.j*dk*z) )
    dAi_dz = -1.j*ki/8*(Ai*( abs(Ai)**2 + 2*abs(Ap)**2 + 2*abs(As)**2) +
                                conj(As)*Ap**2*exp(1.j*dk*z) )
    return array((dAp_dz, dAs_dz, dAi_dz))
    
@njit
def __coupled_eq_3w(kp:float, ks:float, ki:float, dk:float, Idc:float, z:float, A:ndarray) -> ndarray:
    """TWPA coupled wave equations for 3-wave mode according to:
    J. C. Longden and B-K Tan 2024 Eng. Res. Express 6 015068.
    
    Arguments:
    kp -- the pump wavenumber
    ks -- the signal wavenumber
    ki -- the idler wavenumber
    dk -- the wavenumbers relation for 4-w mode: dk = ks + ki - kp
    z -- the coordinate along the amplifier in meters
    Idc -- the DC current expressed in relative current units defined I/I*
    A -- the vector of harmonics amplitudes at the coordinate z: A[0] - pump, A[1] - signal, A[2] - idler
        The amplitudes are expressed in relative current units defined as I/I*. 
    """
    Ap = A[0]
    As = A[1]
    Ai = A[2]
    
    dAp_dz = -1.j*kp/8*(Ap*( abs(Ap)**2 + 2*abs(As)**2 + 2*abs(Ai)**2 + 2*Idc**2) +
                                2*Ai*As*Idc*exp(-1.j*dk*z) )
    dAs_dz = -1.j*ks/8*(As*( abs(As)**2 + 2*abs(Ai)**2 + 2*abs(Ap)**2 + 2*Idc**2) +
                                2*conj(Ai)*Ap*Idc*exp(1.j*dk*z) )
    dAi_dz = -1.j*ki/8*(Ai*( abs(Ai)**2 + 2*abs(Ap)**2 + 2*abs(As)**2 + 2*Idc**2) +
                                2*conj(As)*Ap*Idc*exp(1.j*dk*z) )
    return array((dAp_dz, dAs_dz, dAi_dz))
    
class Gain4w:
    """4-wave gain model for kinetic inductance TWPA.
        
    Attributes:
    l -- the amplifier length in meters
    k_func -- the dispersion function k(f) of the amplifier wavenumber frequency dependence
    As -- signal amplitude at the input of the amplifier expressed in relative current units I/I*. 
        Should be much smaller then the pump amplitude Ap0.
    rtol -- relative tolerance of the IVP solver. Choose small enough that the solution doesn't change anymore.   
    """
    def __init__(self, l:float, k_func:callable):
        self.k_func = k_func
        self.l = l
        
        self.fp = 0
        self.fs = 0
        self.As = 1e-6
        self.rtol = 1e-6
        self._kp = None
        self._ks = None
        self._ki = None
        self._dk = None
        
    def _coupled_eq_4w(self, z, A):
        """Coupled equations wrapper"""
        return _coupled_eq_4w(self._kp, self._ks, self._ki, self._dk, z, A)

    def _k(self):
        """Calculates wavenumbers"""
        self._kp = self.k_func(self.fp)
        self._ks = self.k_func(self.fs)  
        self._ki = self.k_func(2*self.fp-self.fs)
        self._dk = self._ks + self._ki - 2*self._kp 
        
    def solve(self, fs: float|ndarray, fp:float|ndarray, Ap0: float) -> float|ndarray:
        """Solve coupled equations.
        
        Arguments:
        fs -- signal frequency, can be an array
        fp -- pump frequency, can be an array
        Ap0 -- pump amplitude at the input of the amplifier
        
        Returns:
            power gain in dB
        """
        if not hasattr(fp, '__len__'):
            fp = array((fp,))
        if not hasattr(fs, '__len__'):
            fs = array((fs,))
            
        G = zeros((len(fp), len(fs)))    
            
        for i, self.fp in enumerate(fp):
            print( "Solving for Fp point {0} out of {1}...".format(i, len(fp)), end=  '\r')
            for j, self.fs in enumerate(fs):
                self._k()   
                res = si.solve_ivp(self._coupled_eq_4w, (0, self.l), (Ap0+1.j*0,self.As+1.j*0,0+1.j*0), rtol = self.rtol)
                G[i,j] = log10( abs(res['y'][1][-1]/res['y'][1][0])**2 ) *10
        
        if len(fs)==1 and len(fp)==1:
            return G[0,0]
        elif len(fs)==1 or len(fp)==1:
            return G[0]
        else:
            return G
    

class KineticMicrostrip():
    def __init__(self):
        """Kinetic inductance microstrip line model.
        
        Attributes:
        Rsq -- the resistance per square in ohms of the kinetic inductive material
        Tc -- the superconducting critical temperature of the material
        eps -- the dielectric constant of the microstrip dielectric media
        t -- thickness of the microstrip dielectric media in meters
        c_gap -- the energy gap constant of the kinetic inductive material, default is 1.76"""
        
        self.Rsq = 0.
        self.Tc = 0.
        self.eps = 0.
        self.t = 0.
        self.c_gap = 1.76
        
    def capacitance(self, w) -> float:
        """Capacitance per unit length.
        
        Arguments:
        w -- microstrip line width in meters
        """
        return self.eps*sc.epsilon_0*w/self.t
        
    def inductance(self, w) -> float:
        """Inductance per unit length.
        
        Arguments:
        w -- microstrip line width in meters"""
        return self.Rsq*sc.hbar/(w*c_gap*sc.k*self.Tc*pi)
        
    def ki_inductance(self) -> float:
        """Inductance per square."""
        return self.Rsq*sc.hbar/(c_gap*sc.k*self.Tc*pi)    
    
    def velocity(self) -> float:
        """Propagation velocity in m/s."""
        return 1./sqrt( self.Rsq*sc.hbar/(self.c_gap*sc.k*self.Tc*pi)*self.eps*sc.epsilon_0/self.t )
        
    def period(fgap) -> float:
        """Modulation period in meters for a given forbidden zone frequency fgap in Hz."""
        v = self.velocity()
        return v/(2.*fgap)
    
    def width(self, Z) -> float:
        """Transmission line width for a given impedance Z."""
        return sqrt(self.Rsq*sc.hbar*self.t/(self.eps*sc.epsilon_0*pi*self.c_gap*sc.k*self.Tc))/Z
