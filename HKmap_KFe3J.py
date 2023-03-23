#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: kmatan
"""
import numpy as np
from timeit import default_timer as clock
import magcalc as mc
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle


if __name__ == '__main__':
    # spinwave intensity S(Q,\omega)
    st = clock()
    newcalc = 0

    # CCSF
    S = 1.0 / 2.0
    p = [12.8, -1.23, 0.063 * 12.8, -0.25 * 12.8, 0]

    # KFe3J
    # S = 5.0/2.0  # spin value
    # p = [3.23, 0.11, 0.218, -0.195]  # spin Hamiltonian parameter [J1,J2,Dy,Dz]

    nspins = 3  # number of spins in a unit cell
    E_intv = [4, 6]
    qstep = 0.05

    qsx = np.arange(2, 6 + qstep, qstep) * np.pi
    qsy = np.arange(-2, 2 + qstep, qstep) * np.pi
    q = []
    for i in range(len(qsx)):
        for j in range(len(qsy)):
            qx = qsx[i]
            qy = qsy[j]
            qz = 0
            q1 = [qx, qy, qz]
            q.append(q1)

    print(len(q))

    if newcalc == 1:
        qout, En, Sqwout = mc.calc_Sqw(S, q, p, nspins, 'KFe3J', 'r')
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
    plt.pcolor(X, Y, intMat)
    #plt.pcolormesh(X, Y, intMat, norm=LogNorm(vmin=intMat.min(), vmax=intMat.max()), cmap='PuBu_r')
    et = clock()
    print('Total run-time: ', np.round((et-st)/60, 2), ' mins.')
    plt.show()
    '''plt.pcolormesh(X, Y, intMat, norm=LogNorm(vmin=intMat.min(), vmax=intMat.max()), cmap='PuBu_r')
    plt.xlim([0, 2*np.pi])
    plt.ylim([0, 2*np.pi])
    plt.ylabel('H', fontsize=12)
    plt.yticks(np.arange(0, 2*np.pi, np.pi/4))
    plt.xlabel('K', fontsize=12)
    plt.xticks(np.arange(0, 2*np.pi, np.pi/4))
    plt.title('Spinwaves for KFe$_3$(OH)$_6$(SO$_4$)$_2$')
    plt.colorbar()
    plt.show()

    # write data to a file
    with open('sw_jarosite_Sqw_kx.dat', 'w') as f:
        with redirect_stdout(f):
            for i in range(len(qs)):
                print(qs, Ex, intMat)'''
