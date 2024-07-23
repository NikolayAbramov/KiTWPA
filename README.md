# KiTWPA
Kinetic inductance TWPA simulation
<p><strong>4w_calc.py</strong> - calculates dispersion k(f) for a given 4-wave impedance modulation of the TWPA transmission line. 
Results are saved as a *.txt description and HDF5 data file containing k(f) dependence.</p>
<p><strong>4w_gain_map.py</strong> - calculates power gain map in the coordinates of pump frequency and signal frequency. The calculations are done by means of direct solving of the CME for 4-wave mixing. The calculations rely on the dispersion data, so HDF5 dispersion file must be created first by running 4w_calc.py.</p>
<p><strong>4w_gain.py</strong> - calculates power gain versus signal frequency.</p>
