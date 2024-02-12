#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:56:34 2018

@author: Kit Matan and Pharit Piyawongwatthana

Python code to calculate spin-wave dispersion and scattering intensity.
Translated from Mathematica and Octave codes written by Taku J Sato.
"""
# this file contains spin model; see example
# edit spin_model.py
from numba import jit
import spin_model as sm
import sympy as sp
from sympy import I
from sympy import lambdify
from sympy import Add
import numpy as np
# from numpy import linalg as la
from scipy import linalg as la
import timeit
import sys
import pickle
from multiprocessing import Pool
import multiprocessing as mp
# import symengine as se # SymEngine does not support non-commutative symbols!

def substitute_expr(expr, subs_dict):
    """Substitute the Hamiltonian matrix with the dictionary of substitutions
    inputs: HM is the Hamiltonian matrix
            subs_dict is the dictionary of substitutions
    output: HM_subs is the Hamiltonian matrix after substitution"""
    expr_subs = expr.subs(subs_dict, simultaneous=True)
    return expr_subs

def gen_HM(k, S, params):
    """generate the spin Hamiltonian
    inputs: k is a vector kx,ky,kz 
            S is the spin value
            p contains the spin Hamiltonian parameters.
    output: Hamiltonian matrix and Ud that will be used to calculate scattering intensity
    The Hamiltonian and spin structure are given in a separate file called 'spin_model.py'"""
    apos = sm.atom_pos()
    nspins = len(apos)
    apos_ouc = sm.atom_pos_ouc()
    nspins_ouc = len(apos_ouc)

    # generate boson spin operators in local coordinate system,
    # with Z as quantization axis
    c = sp.symbols('c0:%d' % nspins_ouc, commutative=False)
    cd = sp.symbols('cd0:%d' % nspins_ouc, commutative=False)
    Sabn_local = [sp.Matrix((sp.sqrt(S / 2) * (c[i] + cd[i]),
                         sp.sqrt(S / 2) * (c[i] - cd[i]) / I,
                         S - cd[i] * c[i])) for i in range(nspins_ouc)]

    # rotate spin operators to global coordinates
    mp = sm.mpr(params)  # the rotation matrices can depend on the Hamiltonian parameters
    Sabn = [mp[j] * Sabn_local[nspins * i + j] for i in range(int(nspins_ouc / nspins)) for j in range(nspins)]

    # generate the spin Hamiltonian
    HM = sm.Hamiltonian(Sabn, params)
    HM = sp.expand(HM)
    HM_S0 = HM.coeff(S, 0)
    # params[-1] is for magnetic field; the last p value needs to be H
    HM = HM_S0.coeff(params[-1]) * params[-1] + HM.coeff(S ** 1.0) * S + HM.coeff(S) * S + HM.coeff(S ** 2) * S ** 2
    HM = sp.expand(HM)

    # perform Fourier transformation
    ck = [sp.Symbol('ck%d' % j, commutative=False) for i in range(int(nspins_ouc / nspins)) for j in range(nspins)]
    ckd = [sp.Symbol('ckd%d' % j, commutative=False) for i in range(int(nspins_ouc / nspins)) for j in range(nspins)]
    cmk = [sp.Symbol('cmk%d' % j, commutative=False) for i in range(int(nspins_ouc / nspins)) for j in range(nspins)]
    cmkd = [sp.Symbol('cmkd%d' % j, commutative=False) for i in range(int(nspins_ouc / nspins)) for j in range(nspins)]

    # generate dictionary for substitution
    Jex = sm.spin_interactions(params)[0]
    fourier_dict = []
    for i in range(nspins):
        for j in range(nspins_ouc):
            if Jex[i, j] != 0:
                dr = apos[i] - apos_ouc[j]
                k_dot_dr = k[0] * dr[0, 0] + k[1] * dr[0, 1] + k[2] * dr[0, 2]
                ent1 = [cd[i] * cd[j], 1 / 2 * (ckd[i] * cmkd[j] * sp.exp(-I * k_dot_dr).rewrite(sp.sin) +
                                                cmkd[i] * ckd[j] * sp.exp(I * k_dot_dr).rewrite(sp.sin))]
                ent2 = [c[i] * c[j], 1 / 2 * (ck[i] * cmk[j] * sp.exp(I * k_dot_dr).rewrite(sp.sin) +
                                              cmk[i] * ck[j] * sp.exp(-I * k_dot_dr).rewrite(sp.sin))]
                ent3 = [cd[i] * c[j], 1 / 2 * (ckd[i] * ck[j] * sp.exp(-I * k_dot_dr).rewrite(sp.sin) +
                                               cmkd[i] * cmk[j] * sp.exp(I * k_dot_dr).rewrite(sp.sin))]
                ent4 = [c[i] * cd[j], 1 / 2 * (ck[i] * ckd[j] * sp.exp(I * k_dot_dr).rewrite(sp.sin) +
                                               cmk[i] * cmkd[j] * sp.exp(-I * k_dot_dr).rewrite(sp.sin))]
                ent5 = [cd[j] * c[j], 1 / 2 * (ckd[j] * ck[j] + cmkd[j] * cmk[j])]
                fourier_dict.extend([ent1, ent2, ent3, ent4, ent5])

    # substitution Fourier transform
    print('Running the Fourier transform ...')
    print('A number of entries for substitution: ', len(fourier_dict))
    st = timeit.default_timer()
    HM_terms = HM.as_ordered_terms()
    length = len(HM_terms)
    print('A number of terms in the Hamiltonian: ', length)    
    with Pool() as p:
        HMk_terms = p.starmap(substitute_expr, [(expr, fourier_dict) for expr in HM_terms])
    # HMk = HM.subs(fourier_dict, simultaneous=True)
    HMk = Add(*HMk_terms)
    et = timeit.default_timer()
    print('Run-time for the Fourier transform ', np.round((et - st) / 60, 2), ' min.')

    # apply commutation relations so that all 2nd order terms are written as
    # ckd*ck, cmk*cmkd, cmk*ck and ckd*cmkd
    comm_dict = []
    for i in range(nspins):
        for j in range(nspins):
            if i == j:
                comm1 = [ck[i] * ckd[j], ckd[j] * ck[i] + 1]
                comm2 = [cmkd[i] * cmk[j], cmk[j] * cmkd[i] + 1]
                comm_dict.append(comm1)
                comm_dict.append(comm2)
            elif i != j:
                comm3 = [ck[i] * ckd[j], ckd[j] * ck[i]]
                comm4 = [cmkd[i] * cmk[j], cmk[j] * cmkd[i]]
                comm_dict.append(comm3)
                comm_dict.append(comm4)
            comm5 = [ck[i] * cmk[j], cmk[j] * ck[i]]
            comm6 = [cmkd[i] * ckd[j], ckd[j] * cmkd[i]]
            comm_dict.append(comm5)
            comm_dict.append(comm6)

    # HMk_comm is the Hamiltonian after applying the commutation relations
    st = timeit.default_timer()
    print('A number of entries for commutation-relation substitution: ', len(comm_dict))
    HMk_terms = HMk.as_ordered_terms()
    length = len(HMk_terms)
    print('A number of terms in the Hamiltonian: ', length)
    with Pool() as p:
        HMk_terms = p.starmap(substitute_expr, [(expr, comm_dict) for expr in HMk_terms])
    # HMk = HM.subs(fourier_dict, simultaneous=True)
    HMk_comm = Add(*HMk_terms)
    # HMk_comm = HMk.subs(comm_dict, simultaneous=True)
    HMk_comm = HMk_comm.expand()
    et = timeit.default_timer()
    print('Run-time for the commutation relation substitution ', np.round((et - st) / 60, 2), ' min.')

    # create operator tables for ck,...,cmkd and ckd,...,cmk
    # ck,...,cmk are non-commutative but coeff() only supports commutative
    # variables; replace their products by XdXi, which are commutative.
    X = ck[:nspins] + cmkd[:nspins]
    Xd = ckd[:nspins] + cmk[:nspins]
    XdX = []
    XdX_subs = []
    for i in range(2 * nspins):
        for j in range(2 * nspins):
            XdX1 = sp.Symbol('XdX%d' % (i * 2 * nspins + j), commutative=True)
            XdX2 = [Xd[i] * X[j], XdX1]
            XdX_subs.append(XdX2)
            XdX.append(XdX1)

    st = timeit.default_timer()
    print('A number of entries for XdX substitution: ', len(XdX_subs))
    HMk_comm_terms = HMk_comm.as_ordered_terms()
    length = len(HMk_comm_terms)
    print('A number of terms in the Hamiltonian: ', length)
    with Pool() as p:
        HMk_comm_terms = p.starmap(substitute_expr, [(expr, XdX_subs) for expr in HMk_comm_terms])
    HMk_comm_XdX = Add(*HMk_comm_terms)
    # HMk_comm_XdX = HMk_comm.subs(XdX_subs, simultaneous=True)
    et = timeit.default_timer()
    print('Run-time for the substitution of XdX', np.round((et - st) / 60, 2), ' min.')

    # extract the coefficients in front of the 2nd order terms
    H2_matrix_elements = sp.Matrix.zeros(1, len(XdX))
    for i in range(len(XdX)):
        H2_matrix_elements[i] = HMk_comm_XdX.coeff(XdX[i])

    # create a matrix from a list of coefficients and the matrix "g"
    H2 = sp.Matrix(2 * nspins, 2 * nspins, H2_matrix_elements)
    g = np.bmat([[np.eye(nspins), np.zeros((nspins, nspins))],
                 [np.zeros((nspins, nspins)), -np.eye(nspins)]])
    TwogH2 = 2 * g * H2

    # create Ud matrix, spin rotation operators
    # note that Ud can contain Hamiltonian parameters so we need to use sympy
    Ud_row = sp.Matrix.zeros(3, 3)
    Ud = sp.Matrix.zeros(3, 3 * nspins)
    for i in range(nspins):
        for j in range(nspins):
            if j == 0:
                if i == 0:
                    Ud_row = mp[i]
                else:
                    Ud_row = sp.Matrix.zeros(3, 3)
            else:
                if i == j:
                    Ud_row = Ud_row.col_insert(3 * nspins + 1, mp[i])
                else:
                    Ud_row = Ud_row.col_insert(3 * nspins + 1, sp.Matrix.zeros(3, 3))
        if i == 0:
            Ud = Ud_row
        else:
            Ud = Ud.row_insert(3 * nspins + 1, Ud_row)

    return TwogH2, Ud


def gram_schmidt(x):
    q, r = np.linalg.qr(x)
    return q


def KKdMatrix(Sp, Hkp, Hkm, Ud, q, nspins):
    """Calculate K and Kd matrices
        Inputs:
            Sp: spin operators
            Hkp: Hamiltonian for positive q
            Hkm: Hamiltonian for negative q
            Ud: spin rotation operators
            q: momentum transfer
            nspins: number of spins
        outputs:
            K: K matrix
            Kd: Kd matrix
            e_val: eigenvalues"""
    dEdeg = 10e-12  # degeneracy threshold
    G = np.bmat([[np.eye(nspins), np.zeros((nspins, nspins))],
                 [np.zeros((nspins, nspins)), -np.eye(nspins)]])

    # positive q  [+q]
    w, v = la.eig(Hkp)  # e_val, e_vec
    es = w.argsort()  # sorting e_val from lowest to highest
    vtmp1 = v[:, es][:, nspins:2 * nspins]
    etmp1 = w[es][nspins:2 * nspins]
    vtmp2 = v[:, es][:, 0:nspins]
    etmp2 = w[es][0:nspins]
    ess = (np.abs(etmp2)).argsort()
    vtmp3 = vtmp2[:, ess]
    etmp3 = etmp2[ess]
    e_val = np.r_[etmp1, etmp3]
    e_vec = np.hstack((vtmp1, vtmp3))

    # Gram-Schmidt Orthogonalization [+q] '''
    ndeg = 0
    for i in range(1, 2 * nspins):
        if abs(e_val[i] - e_val[i - 1]) < dEdeg:
            ndeg = ndeg + 1
        elif ndeg > 0:
            vtmp1 = e_vec[:, i - ndeg - 1:i]
            e_vec[:, i - ndeg - 1:i] = gram_schmidt(vtmp1)
            ndeg = 0
    if ndeg > 0:
        vtmp1 = e_vec[:, 2 * nspins - ndeg:2 * nspins]
        e_vec[:, 2 * nspins - ndeg:2 * nspins] = gram_schmidt(vtmp1)

    # Determine Alpha matrix [+q]
    Td = la.inv(e_vec)
    al = np.sqrt(np.abs(np.real(Td @ G @ np.conj(Td.T))))
    al[abs(al) < 1e-6] = 0  # truncate small values

    # prepare upper-lower-swapped and c.c.ed evec for later purpose
    evecswap = np.conj(np.vstack((e_vec[nspins:2 * nspins, :],
                                  e_vec[0:nspins, :])))

    # diagonalize Hkm for negative q  [-q] '''
    wm, vm = la.eig(Hkm)  # e_val, e_vec
    esm = wm.argsort()  # sorting e_val from lowest to highest
    vtmpm1 = vm[:, esm][:, nspins:2 * nspins]
    etmpm1 = wm[esm][nspins:2 * nspins]
    vtmpm2 = vm[:, esm][:, 0:nspins]
    etmpm2 = wm[esm][0:nspins]
    essm = (abs(etmpm2)).argsort()[::1]
    vtmpm3 = vtmpm2[:, essm]
    etmpm3 = etmpm2[essm]
    e_valm = np.r_[etmpm1, etmpm3]
    e_vecm = np.hstack((vtmpm1, vtmpm3))

    # find degeneracy and Gram-Schmidt Orthogonalization [-q]
    torr = 1e-5
    ndeg = 0
    for i in range(2 * nspins):
        if np.abs(e_valm[np.mod(i, 2 * nspins - 1) + 1] - e_valm[i]) < dEdeg:
            ndeg = ndeg + 1
        elif ndeg > 0:
            vtmpm1 = e_vecm[:, i - ndeg:i + 1]
            if i < nspins:
                tmpsum = np.zeros(nspins)
                tmpevecswap = evecswap[:, nspins:2 * nspins]
                for j in range(ndeg):
                    # this does not work properly if change @ to *
                    tmpsum = tmpsum + (np.conj(vtmpm1[:, j]).T @ tmpevecswap)
                tmpsum_abs = np.abs(np.array(tmpsum))
                newevec = tmpevecswap[:, tmpsum_abs > torr]
                if newevec.shape[1] != ndeg + 1:
                    print('A number of degenerate eigenvectors is different from degeneracy; something is strange.')
                    print(q, '\t', i - ndeg + 1, ':', i + 1, '\t', tmpsum_abs)
                    print(q, '\t', i - ndeg + 1, ':', i + 1, '\t', np.real(e_valm))
                    print('No. col: ', newevec.shape[1], ' BUT degeneracy: ', ndeg + 1)
                    print(newevec.shape)
                    print('The program will exit ...')
                    sys.exit()
                else:
                    pass
            else:  # if i > nspins
                tmpsum = np.zeros(nspins)
                tmpevecswap = evecswap[:, 0:nspins]
                for j in range(ndeg):
                    tmpsum = tmpsum + (np.conj(vtmpm1[:, j]).T @ tmpevecswap)
                tmpsum_abs = np.abs(np.array(tmpsum))
                newevec = tmpevecswap[:, tmpsum_abs > torr]
                if newevec.shape[1] != ndeg + 1:
                    print('A number of degenerate eigenvectors is different from degeneracy; something is strange.')
                    print(q, '\t', i - ndeg + 1, ':', i + 1, '\t', tmpsum_abs)
                    print(q, '\t', i - ndeg + 1, ':', i + 1, '\t', np.real(e_valm))
                    print('No. col: ', newevec.shape[1], ' BUT degeneracy: ', ndeg + 1)
                    print(newevec.shape)
                    print('The program will exit ...')
                    sys.exit()
                else:
                    pass

            e_vecm[:, i - ndeg:i + 1] = newevec
            ndeg = 0

    # Determine Alpha matrix [-q]
    Tdm = la.inv(e_vecm)
    alm = np.sqrt(np.abs(np.real(Tdm @ G @ np.conj(Tdm.T))))
    alm = alm.astype(complex)
    alm[abs(alm) < 1e-6] = 0  # truncate small values

    # prepare upper-lower-swapped and c.c.ed evec for later purpose
    evecmswap = np.conj(np.vstack((e_vecm[nspins:2 * nspins, :],
                                   e_vecm[0:nspins, :])))
    Ntmpm = 0
    tmpm = np.zeros((e_vecm.shape[0], e_vecm.shape[1]), dtype=complex)
    tmpevalm = np.zeros(e_valm.size, dtype=complex)

    for i in range(nspins):
        for j in range(nspins, 2 * nspins):
            if np.abs(np.dot(np.conj((e_vec[:, i]).T), evecmswap[:, j])) > \
                    np.sqrt(np.dot(np.conj((e_vec[:, i]).T), e_vec[:, i]) *
                            np.dot(np.conj((evecmswap[:, j]).T), evecmswap[:, j])) - 1.0e-5:  # if parallel
                tmpm[:, i + nspins] = e_vecm[:, j]
                tmpevalm[i + nspins] = e_valm[j]
                alm[i + nspins, i + nspins] = \
                    np.conj(al[i, i] * np.divide(e_vec[np.abs(e_vec[:, i]) > 1e-5, i],
                                                 evecmswap[np.abs(evecmswap[:, j]) > 1e-5, j])[0])
                Ntmpm = Ntmpm + 1
    for i in range(nspins, 2 * nspins):
        for j in range(nspins):
            if np.abs(np.dot(np.conj((e_vec[:, i]).T), evecmswap[:, j])) > \
                    np.sqrt(np.dot(np.conj((e_vec[:, i]).T), e_vec[:, i]) *
                            np.dot(np.conj((evecmswap[:, j]).T), evecmswap[:, j])) - 1.0e-5:
                tmpm[:, i - nspins] = e_vecm[:, j]
                tmpevalm[i - nspins] = e_valm[j]
                alm[i - nspins, i - nspins] = \
                    np.conj(al[i, i] * np.divide(e_vec[np.abs(e_vec[:, i]) > 1e-5, i],
                                                 evecmswap[np.abs(evecmswap[:, j]) > 1e-5, j])[0])
                Ntmpm = Ntmpm + 1
    e_vecm = tmpm
    # e_valm = tmpevalm  # uncomment this if you want to use eigenvalues

    # now we have inverse of T which relates Y to X as X = invT Y
    invT = e_vec * al
    invTm = e_vecm * alm
    # T = la.inv(invT)
    # Tm = la.inv(invTm)

    Udd = np.zeros((3 * nspins, 2 * nspins), dtype=complex)
    for i in range(0, nspins):
        Udd[3 * i, i] = 1
        Udd[3 * i, i + nspins] = 1
        Udd[3 * i + 1, i] = 1 / I
        Udd[3 * i + 1, i + nspins] = -1 / I

    # S is previously defined as symbolic so here we use Sp instead
    S = Sp
    K = (np.sqrt(2 * S) / 2) * Ud @ Udd @ invT
    Kd = (np.sqrt(2 * S) / 2) * Ud @ Udd @ invTm
    K[abs(K) < 1e-6] = 0  # truncate small values
    Kd[abs(Kd) < 1e-6] = 0  # truncate small values

    return K, Kd, e_val


def process_calc_Sqw(HMat, Ud, k, q, nspins, Sp):
    """Process to calculate scattering intensity that will be called by Pool to run using multiprocessing
        Inputs:
            HMat: Hamiltonian matrix
            Ud: unitary matrix
            k: wave vector
            q: momentum transfer
            nspins: number of spins
            Sp: spin quantum number
        Outputs:
            Sqwout: scattering intensity
            qout: momentum transfer
            En: eigenvalues"""
    Sqwout0 = np.zeros(nspins, dtype=complex)
    # Create a lambda function for HMat
    HMat_func = lambdify([k[0], k[1], k[2]], HMat)
    # Use the lambda function to substitute and evaluate
    Hkp = HMat_func(q[0], q[1], q[2])
    Hkp = np.mat(Hkp).astype(np.complex_)
    Hkm = HMat_func(-q[0], -q[1], -q[2])
    Hkm = np.mat(Hkm).astype(np.complex_)

    K, Kd, evals = KKdMatrix(Sp, Hkp, Hkm, Ud, q, nspins)
    # En = np.real_if_close(evals[0:nspins])
    En = np.real(evals[0:nspins])
    qout = q
    for l in range(nspins):  # calculate all modes
        SS = np.zeros((3, 3), dtype=complex)  # xyz
        Sqw = 0  # scattering cross-section (polarization factor incl.)
        if np.sqrt(np.dot(q, q)) < 1e-5:
            qt = [0, 0, 0]
        else:
            qt = q / np.sqrt(np.dot(q, q))
        for alpha in range(3):  # xyz
            for beta in range(3):  # xyz
                if alpha == beta:
                    d = 1.0
                else:
                    d = 0.0
                for i in range(nspins):
                    for j in range(nspins):
                        SS[alpha, beta] = SS[alpha, beta] + \
                                          K[3 * (i - 1) + alpha, l] * \
                                          Kd[3 * (j - 1) + beta, l + nspins]
                Sqw = Sqw + (d - qt[alpha] * qt[beta]) * SS[alpha, beta]
        if np.abs(np.imag(Sqw)) > 1e-5:
            print('Error: imaginary part of Sqw is finite (> 1e-5).')
            print(q, Sqw)
        Sqwout0[l] = Sqwout0[l] + Sqw  # Sqwout is the final output matrix
    # Sqwout = np.real_if_close(Sqwout0)
    Sqwout = np.real(Sqwout0)

    return qout, En, Sqwout


def calc_Sqw(Sp, q, p, nspins, file, rd_or_wr):
    """calculate the scattering intensity; Sqw with the geometric factor
        Inputs:
            Sp: spin quantum number
            q: momentum transfer
            p: parameters
            nspins: number of spins
            file: file name
            rd_or_wr: read or write the Hamiltonian matrix to a file
        Outputs:
            qout: momentum transfer
            En: energy
            Sqwout: scattering intensity"""
    print('Calculating scattering intensity ...')
    kx, ky, kz = sp.symbols('kx ky kz', real=True)
    k = [kx, ky, kz]
    S = sp.Symbol('S', real=True)
    params = sp.symbols('p0:%d' % len(p), real=True)
    # call gen_HM to create the matrix to calculate spin-waves
    # or read the matrix from a file previously saved
    if rd_or_wr == 'w':
        print('Generating the matrix ...')
        HMat, Ud = gen_HM(k, S, params)  # this function takes a very long time!!!
        # write Hamiltonian to a file 
        with open('pckFiles/' + file + '_HM.pck', 'wb') as outHM:
            outHM.write(pickle.dumps(HMat))
        with open('pckFiles/' + file + '_Ud.pck', 'wb') as outUd:
            outUd.write(pickle.dumps(Ud))
    elif rd_or_wr == 'r':
        # read Hamiltonian from a file
        print('Reading the matrix from a file ...')
        with open('pckFiles/' + file + '_HM.pck', 'rb') as inHM:
            HMat = pickle.loads(inHM.read())
        with open('pckFiles/' + file + '_Ud.pck', 'rb') as inUd:
            Ud = pickle.loads(inUd.read())
    else:
        print('Does not recognize the input ' + rd_or_wr + '.')
        sys.exit()

    # substitute for Hamiltonian parameters and S
    param_subs = [[S, Sp]] + [[params[i], p[i]] for i in range(len(p))]
    HMat = HMat.subs(param_subs, simultaneous=True).evalf()
    Ud = Ud.subs(param_subs, simultaneous=True).evalf()
    Ud = np.mat(Ud).astype(np.float_)

    print('Running the diagonalization ...')
    st = timeit.default_timer()
    # generate arguments for multiprocessing input
    args = [(HMat, Ud, k, q[i], nspins, Sp) for i in range(len(q))]
    # use multiprocessing
    with Pool() as pool:
        results = pool.starmap(process_calc_Sqw, args)

    qout, En, Sqwout = zip(*results)

    et = timeit.default_timer()
    print('Run-time for the diagonalization: ', np.round((et - st) / 60, 2))

    return qout, En, Sqwout


def process_calc_disp(HMat, k, q, nspins):
    """Process to calculate dispersion relation that will be called by Pool to run using multi-processing
        Inputs:
            HMat: Hamiltonian matrix
            k: momentum
            q: momentum transfer
            nspins: number of spins
        Outputs:
            eigval: eigenvalues of the Hamiltonian matrix"""
    # HMat_k = HMat.subs({k[0]: q[0], k[1]: q[1], k[2]: q[2]}, simultaneous=True).evalf()
    # Create a lambda function for HMat
    HMat_func = lambdify([k[0], k[1], k[2]], HMat)
    # Use the lambda function to substitute and evaluate
    HMat_k = HMat_func(q[0], q[1], q[2])
    HMat_k = np.mat(HMat_k).astype(np.complex_)
    eigval = la.eigvals(HMat_k)
    # check whether all eigen-energies are real
    for i in range(nspins):
        if np.abs(np.imag(eigval[i])) > 1e-5:
            print('Error: imaginary part of Sqw is finite (> 1e-5).')
            print(q, eigval[i])
    # eigval = np.real_if_close(sorted(eigval))[nspins:]
    eigval = np.real(sorted(eigval))[nspins:]
    return eigval


def calc_disp(Sp, q, p, nspins, file, rd_or_wr):
    """Calculate dispersion relation
        Inputs:
            Sp: spin quantum number
            q: momentum transfer
            p: parameters
            nspins: number of spins
            file: file name
            rd_or_wr: read or write the Hamiltonian matrix to a file
        Outputs:
            En: energy"""
    kx, ky, kz = sp.symbols('kx ky kz', real=True)
    k = [kx, ky, kz]
    S = sp.Symbol('S', real=True)
    params = sp.symbols('p0:%d' % len(p), real=True)
    # call gen_HM to create the matrix to calculate spin-waves
    # or read the matrix from a file previously saved
    if rd_or_wr == 'w':
        print('Generating the matrix ...')
        HMat, Ud = gen_HM(k, S, params)
        # write Hamiltonian to a file 
        with open('pckFiles/' + file + '_HM.pck', 'wb') as outHM:
            outHM.write(pickle.dumps(HMat))
        with open('pckFiles/' + file + '_Ud.pck', 'wb') as outUd:
            outUd.write(pickle.dumps(Ud))
    elif rd_or_wr == 'r':
        # read Hamiltonian from a file
        print('Reading the matrix from a file ...')
        with open('pckFiles/' + file + '_HM.pck', 'rb') as inHM:
            HMat = pickle.loads(inHM.read())
    else:
        print('Wrong input!')
        sys.exit()

    # create a substitution list for the Hamiltonian parameters
    param_subs = [[S, Sp]] + [[params[i], p[i]] for i in range(len(p))]
    HMat = HMat.subs(param_subs, simultaneous=True).evalf()

    print('Running the diagonalization ...')
    st = timeit.default_timer()
    # generate arguments for multiprocessing input
    args = [(HMat, k, q_i, nspins) for q_i in q]
    # use multiprocessing
    with Pool() as pool:
        En = pool.starmap(process_calc_disp, args)

    et = timeit.default_timer()
    print('Run-time for the diagonalization: ', np.round((et - st) / 60, 2))

    return En
