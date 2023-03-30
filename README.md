# magcalc
Calculate spin-wave excitations based on the linear spin-wave theory.  The program is written in Python using SymPy.  It was used to calcualte spin-wave exctaitons for the kagome lattice antiferromagnet **KFe<sub>3</sub>(OH)<sub>6</sub>(SO<sub>4</sub>)<sub>2</sub>** and for the non-reciprocal magnons in **&alpha;-Cu<sub>2</sub>V<sub>2</sub>O<sub>7</sub>**.

### Requirement:
  - **Python3**
  - SymPy
  - NumPy
  - Matplotlib
  - lmfit


### File description:
  - **magcalc.py**: calculate the spin-wave disersion and scattering intensity
  - **disp_...py**: calcualte and plot the spin-wave dispersion
  - **EQmap_...py**: calcualte and plot the intensity map as a function of energy and momentum
  - **HKmap_...py**: calcualte and plot the intensity map as a function of momenta
  - **lmfit_...py**: fit the dipersion
  - **spin_model.py**: contains the information about the spin model used
to calculate spin-waves by **magcalc.py**
  - **data** (folder): contain the neutron scattering data for KFe<sub>3</sub>(OH)<sub>6</sub>(SO<sub>4</sub>)<sub>2</sub> and &alpha;-Cu<sub>2</sub>V<sub>2</sub>O<sub>7</sub>
  - **pckFiles** (folder): contain auxillary files used to stored matrices and calculated intensity

### Path:
In the terminal, set up the following path
```
$ export PYTHONPATH=<Path to magcalc.py>
```
and then you can run, for example,
```
$ python3 KFe3J/disp_KFe3J.py
```
### Issues:
  - The code uses SymPy for symbolic manipulation and it can be very slow for a large system.  For example, it takes about 1 hour to generate a matrix for **&alpha;-Cu<sub>2</sub>V<sub>2</sub>O<sub>7</sub>** with 16 spins in a magnetic unit cell.
  - One has to re-edit **spin_model.py** for a different system and it is not straightforward to work with it.
