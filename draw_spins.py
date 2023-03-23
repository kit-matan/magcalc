#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 2018

@author: Kit Matan

Plot the spin structure
"""

# import magcalc as mc
import matplotlib.pyplot as plt
import spin_model as sm
from scipy import linalg as la
import numpy as np
# from mpl_toolkits.mplot3d import axes3d

if __name__ == "__main__":
    apos_ouc = sm.atom_pos_ouc()
    apos = sm.atom_pos()
    p = [1.0, 2.0, 0.2, 0.2, 0.0]  # CCSF_P21n
    Jex = sm.spin_interactions(p)[0]
    spin_rot_mat = sm.mpr(p)
    spin_vector_list = []
    for srm in spin_rot_mat:
        spin_vector = srm * np.matrix([[0], [0], [1]])
        spin_vector_list.append(spin_vector)
    X_ouc = np.zeros(len(apos_ouc))
    Y_ouc = np.zeros(len(apos_ouc))
    Z_ouc = np.zeros(len(apos_ouc))
    U_ouc = np.zeros(len(apos_ouc))
    V_ouc = np.zeros(len(apos_ouc))
    W_ouc = np.zeros(len(apos_ouc))
    C_ouc = np.zeros(len(apos_ouc))
    for i in range(len(apos_ouc)):
        X_ouc[i] = apos_ouc[i][0, 0]
        Y_ouc[i] = apos_ouc[i][0, 1]
        Z_ouc[i] = apos_ouc[i][0, 2]
        U_ouc[i] = spin_vector_list[i % len(apos)][0, 0]
        V_ouc[i] = spin_vector_list[i % len(apos)][1, 0]
        W_ouc[i] = spin_vector_list[i % len(apos)][2, 0]
        if i < len(apos):
            C_ouc[i] = 0
        else:
            C_ouc[i] = 1
    X_uc = X_ouc[0:len(apos)]
    Y_uc = Y_ouc[0:len(apos)]
    Z_uc = Z_ouc[0:len(apos)]
    U_uc = U_ouc[0:len(apos)]
    V_uc = V_ouc[0:len(apos)]
    W_uc = W_ouc[0:len(apos)]
    C_uc = C_ouc[0:len(apos)]
    shortest_bond = min(la.norm([X_uc[0] - X_uc[1], Y_uc[0] - Y_uc[1], 0]), la.norm([X_uc[0] - X_uc[2], Y_uc[0] - Y_uc[2], 0]))
    plt.subplot(121)
    for i in range(len(apos)):
        for j in range(len(apos_ouc)):
            if Jex[i,j] == p[0]:
                plt.plot([X_ouc[i], X_ouc[j]], [Y_ouc[i], Y_ouc[j]], 'r:', zorder=1)
            else:
                continue
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    plt.quiver(X_ouc, Y_ouc, U_ouc, V_ouc, C_ouc, pivot='mid', cmap=plt.cm.copper, scale=8, width=0.02, zorder=2)
    # ax.quiver(X_ouc, Y_ouc, Z_ouc, U_ouc, V_ouc, W_ouc, C_ouc, length=0.1, normalize=True)
    # plt.quiver(Y_uc, Z_uc, V_uc, W_uc, C_uc, pivot='mid', cmap=plt.cm.copper, scale=4, width=0.02, zorder=2)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    for i in range(len(apos)):
        for j in range(len(apos_ouc)):
            if Jex[i,j] == p[0]:
                plt.plot([X_ouc[i], X_ouc[j]], [Y_ouc[i], Y_ouc[j]], 'r:', zorder=1)
            else:
                continue
    plt.quiver(X_uc, Y_uc, U_uc, V_uc, C_uc, pivot='mid', cmap=plt.cm.copper, scale=4, width=0.02, zorder=2)
    plt.axis('equal')
    plt.xlim(min(X_uc) - shortest_bond, max(X_uc) + shortest_bond)
    plt.ylim(min(Y_uc) - shortest_bond, max(Y_uc) + shortest_bond)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('figures/spin_structure_CCSF_P21n.eps', format='eps', dpi=1000)
    plt.show()
