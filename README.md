# magcalc
Calculate spin-wave excitations based on the linear spin-wave theory.  The program is written in Python using SymPy.  It was used to calcualte spin-wave exctaitons for the kagome lattice antiferromagnet KFe<sub>3</sub>(OH)<sub>6</sub>(SO<sub>4</sub>)<sub>2</sub> and for the non-reciprocal magnons in &alpha;-Cu<sub>2</sub>V<sub>2</sub>O<sub>7</sub>.

### Requirement
<ul>
  <li>Python3</li>
  <li>SymPy</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
  <li>lmfit</li>
</ul>

### File description
<ul>
  <li>magcalc.py: calculate the spin-wave disersion and scattering intensity</li>
  <li>disp_....py: calcualte and plot the spin-wave dispersion</li> 
  <li>EQmap_....py: calcualte and plot the intensity map as a function of energy and momentum</li>
  <li>HKmap_....py: calcualte and plot the intensity map as a function of momenta</li>
  <li>lmfit_....py: fit the dipersion</li>
  <li>data (folder): contain the neutron scattering data for KFe<sub>3</sub>(OH)<sub>6</sub>(SO<sub>4</sub>)<sub>2</sub> and &alpha;-Cu<sub>2</sub>V<sub>2</sub>O<sub>7</sub></li>
  <li>pckFiles (folder): contain auxillary files used to stored matrices and calculated intensity</li>
</ul>
