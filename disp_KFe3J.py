#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:18:18 2018

@author: Kit Matan

Calculate and plot the spin-wave dispersion for KFe3(OH)6(SO4)2
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import magcalc as mc

if __name__ == "__main__":
    st = default_timer()
    # S = 5.0 / 2.0  # spin value
    # positive chirality
    # p = [3.23, 0.11, 0.218, -0.195, 0]  # spin Hamiltonian parameter [J1, J2, Dy, Dz, H]
    # negative chirality
    # p = [3.23, -0.11, 0.1, 0.5, 0]  # spin Hamiltonian parameter [J1, J2, Dy, Dz, H]
    # S = 1.0 / 2.0  # spin value
    # p = [13.6, -1.07, 0.057 * 13.6, -0.29 * 13.6, 0]  # spin Hamiltonian parameter [J11,J12,J13,J2,Dy,Dz]
    nspins = 3  # number of spins in a unit cell

    # CCSF
    S = 1.0 / 2.0
    p = [12.0, 0.2, 0.063 * 12.0, -0.25 * 12.0, 0]
    # p = [12.8, -1.23, 0.063 * 12.8, -0.25 * 12.8, 0]
    qsx = np.arange(0, 2 * np.pi / np.sqrt(3) + 0.05, 0.05)
    qsy = np.arange(0, 2 * np.pi + 0.05, 0.05)
    q = []
    for i in range(len(qsx)):
        qx = qsx[i]
        qy = 0
        qz = 0
        q1 = np.array([qx, qy, qz])
        q.append(q1)
    for i in range(len(qsy)):
        qx = 0
        qy = qsy[i]
        qz = 0
        q1 = np.array([qx, qy, qz])
        q.append(q1)
    En = mc.calc_disp(S, q, p, nspins, 'KFe3J', 'w')

    Ekx1 = [En[i][0] for i in range(len(qsx))]
    Ekx2 = [En[i][1] for i in range(len(qsx))]
    Ekx3 = [En[i][2] for i in range(len(qsx))]
    Eky1 = [En[len(qsx) + i][0] for i in range(len(qsy))]
    Eky2 = [En[len(qsx) + i][1] for i in range(len(qsy))]
    Eky3 = [En[len(qsx) + i][2] for i in range(len(qsy))]

    # plot the spin-wave dispersion
    qsyn = 2 * np.pi + 2 * np.pi / np.sqrt(3) - qsy
    plt.plot(qsx, Ekx1, 'r-')
    plt.plot(qsx, Ekx2, 'g-')
    plt.plot(qsx, Ekx3, 'b-')
    plt.plot(qsyn, Eky1, 'r-')
    plt.plot(qsyn, Eky2, 'g-')
    plt.plot(qsyn, Eky3, 'b-')
    plt.plot([2 * np.pi / np.sqrt(3), 2 * np.pi / np.sqrt(3)], [-1, 25], 'k:')
    plt.plot([2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3, 2 * np.pi /
              np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3], [-1, 25], 'k:')
    plt.xlim([0, 2 * np.pi / np.sqrt(3) + 2 * np.pi])
    plt.ylim([0, 20])
    plt.xticks([])
    plt.text(0, -1, '$\Gamma$', fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) - 0.1, -1, 'M')
    plt.text(2 * np.pi/np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3-0.1, -1, 'K', fontsize=12)
    plt.text(0, -1, '$\Gamma$', fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) + 2 * np.pi - 0.1, -1, '$\Gamma$', fontsize=12)
    plt.ylabel('$\hbar\omega$ (meV)', fontsize=12)
    plt.yticks(np.arange(0, 21, 5.0))
    plt.title('Spin-waves for KFe$_3$(OH)$_6$(SO$_4$)$_2$')
    et = default_timer()
    print('Total run-time: ', np.round((et-st) / 60, 2), ' min.')
    plt.show()
