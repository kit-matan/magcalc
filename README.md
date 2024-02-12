# pyMagCalc
*pyMagCalc* (**py**thon for **Mag**non **Calc**ulations) calculates spin-wave excitations based on the linear spin-wave theory.  The program is written in Python using SymPy.  It was used to calculate spin-wave excitations in the kagome lattice antiferromagnet **KFe<sub>3</sub>(OH)<sub>6</sub>(SO<sub>4</sub>)<sub>2</sub>**, the non-reciprocal magnons in **&alpha;-Cu<sub>2</sub>V<sub>2</sub>O<sub>7</sub>**, and spin-waves excitations in **Zn-doped Cu<sub>2</sub>V<sub>2</sub>O<sub>7</sub>**, where the non-reciprocal magnons are absent.

### Requirement:
  - **Python3**
    - SymPy
    - NumPy
    - SciPy
    - Matplotlib
    - lmfit


### File Description:
  - **magcalc.py**: calculate the spin-wave dispersion and scattering intensity
  - **disp_---.py**: calculate and plot the spin-wave dispersion
  - **EQmap_---.py**: calculate and plot the intensity map as a function of energy and momentum
  - **HKmap_---.py**: calculate and plot the intensity map as a function of momenta
  - **lmfit_---.py**: fit the dispersion
  - **spin_model.py**: contains the information about the spin model used
to calculate spin-waves by **magcalc.py**
  - **data** (folder): contain the neutron scattering data for KFe<sub>3</sub>(OH)<sub>6</sub>(SO<sub>4</sub>)<sub>2</sub>, &alpha;-Cu<sub>2</sub>V<sub>2</sub>O<sub>7</sub>, and Zn-doped Cu<sub>2</sub>V<sub>2</sub>O<sub>7</sub>


### How to run the program:
In a terminal, run the following commands
```
$ export PYTHONPATH=<Path to magcalc.py>
$ cd <Path to magcalc.py>
$ mkdir pckFiles
```
and then run, for example,
```
$ python3 KFe3J/disp_KFe3J.py
```
to calculate and plot spin-wave dispersion. You must first run **disp_---.py** to generate and store a matrix in a .pck file. The folder **pckFiles** contains auxiliary files used to store matrices and calculated intensity.

One has to edit **spin_model.py** for a different system. 

### Issues:
  - The code uses SymPy for symbolic manipulation and can be very slow for a large system.  For example, it takes about 1 hour (on *iMac 5K 27-inch 2020 3.6GHz 10-Core Intel Core i9 with 32GB RAM*) to generate a matrix for **&alpha;-Cu<sub>2</sub>V<sub>2</sub>O<sub>7</sub>** with 16 spins in a magnetic unit cell.
  - Calculations can be prolonged for a large matrix even with *multiprocessing*.
