# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: Kit Matan
"""
import sys
sys.path.append('../')
import numpy as np
from timeit import default_timer
import magcalc as mc
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle


def plot_hkmap(p, S, nspins, wr, newcalc, E_intv, qstep):
    """Spinwave intensity S(Q,\omega) 2D Q-map
        Inputs:
            p: list of parameters
            S: spin value
            nspins: number of spins in a unit cell
            wr: 'w' for write to file, 'r' for read from file
            newcalc: 1 for new calculation, 0 for read from file
            E_intv: energy interval for integration"""
    qsx = np.arange(0 - qstep / 2, 6 + qstep / 2, qstep) * np.pi
    qsy = np.arange(0 - qstep / 2, 6 + qstep / 2, qstep) * np.pi
    q = []
    for i in range(len(qsx)):
        for j in range(len(qsy)):
            q1 = np.array([qsx[i], qsy[j], 0])
            q.append(q1)

    print(len(q))

    if newcalc == 1:
        qout, En, Sqwout = mc.calc_Sqw(S, q, p, nspins, 'KFe3J', wr)
        with open('pckFiles/KFe3J_HKmap_En.pck', 'wb') as outEn:
            outEn.write(pickle.dumps(En))
        with open('pckFiles/KFe3J_HKmap_Sqw.pck', 'wb') as outSqwout:
            outSqwout.write(pickle.dumps(Sqwout))
    else:
        with open('pckFiles/KFe3J_HKmap_En.pck', 'rb') as inEn:
            En = pickle.loads(inEn.read())
        with open('pckFiles/KFe3J_HKmap_Sqw.pck', 'rb') as inSqwout:
            Sqwout = pickle.loads(inSqwout.read())

    print(len(qsy), len(En), len(Sqwout))
    intMat = np.zeros((len(qsy), len(qsx)))
    for i in range(len(qsy)):
        for j in range(len(qsx)):
            for band in range(len(En[0])):
                if En[i*len(qsx)+j][band] < E_intv[1] and En[i*len(qsx)+j][band] > E_intv[0]:
                    intMat[i, j] = intMat[i, j] + Sqwout[i*len(qsy)+j][band]
                else:
                    intMat[i, j] = intMat[i, j]
    print(qsy.shape, qsx.shape, intMat.shape)
    X, Y = np.meshgrid(qsx / np.pi, qsy / np.pi)
    plt.pcolor(X, Y, intMat, cmap='PuBu_r')
    # plt.pcolormesh(X, Y, intMat, norm=LogNorm(vmin=intMat.min() + 1e-0, vmax=intMat.max()), cmap='PuBu_r')
    plt.title('Spinwave intensity Q-map for KFe$_3$(OH)$_6$(SO$_4$)$_2$')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    st_main = default_timer()
    # KFe3Jarosite
    S = 5.0 / 2.0  # spin value
    # CCSF
    # S = 1.0 / 2.0
    # p = [12.8, -1.23, 0.063 * 12.8, -0.25 * 12.8, 0]
    nspins = 3  # number of spins in a unit cell
    p = [3.23, 0.11, 0.218, -0.195, 0]
    plot_hkmap(p, S, nspins, 'r', 0, [6, 8], 0.1)
    et_main = default_timer()
    print('Total run-time: ', np.round((et_main-st_main) / 60, 2), ' min.')

# %%
