"""
File Name : TP1 Ma223
Maxime, Brieuc, 2PF
"""
import numpy as np
import matplotlib.pyplot as plt
import time as time
import math as m
import os
import csv
import statistics as st

"""
ALGORITHME DE GAUSS
"""
### QUESTION 1 ###

def ReductionGauss(Aaug):
    """
    Rend la matrice obtenue après l'application de 
    la métode de Gauss à une matrice augmentée de format (n, n+1)
    """

    l = len(Aaug)
    c = len(Aaug[0])
    # print(l, c)

    for i in range(0, l-1):
        # print(p)
        for k in range(i+1, l):
            g = Aaug[k][i] / Aaug[i][i]
            # print(g)
            for j in range(i, l):
                Aaug[k][j] = Aaug[k][j] - (g * Aaug[i][j])
            Aaug[k][c-1] = Aaug[k][c-1] - (g * Aaug[i][c-1])

    return Aaug


A = np.array([[2, 5, 6, 7], [4, 11, 9, 12], [-2, -8, 7, 3]])
B = np.array([[1, 1, 1, 1, 1], [2, 4, -3, 2, 1], [-1, -1, 0, -3, 2], [1, -1, 4, 9, -8]])

# print(ReductionGauss(A))
# print(ReductionGauss(B))


### QUESTION 2 ###

def ResolutionSystTriSup(Taug):
    """
    Résolution du système triangulaire obtenue avec la fonction précédente
    """
    l = len(Taug)
    c = len(Taug[0])
    x = [0]*l
    x[l-1] = Taug[l-1][c-1] / Taug[l-1][c-2]
    
    for i in range(l-2, -1, -1):
        x[i] = Taug[i][c-1]
        for j in range(i+1, l):
            x[i] = x[i] - Taug[i][j]*x[j]
        x[i] = (x[i]/Taug[i][i])
    return x



### QUESTION 3 ###

def Gauss(A, B):
    """
    Application de la méthode de Gauss pour résoudre le système (sans pivot)
    """
    if len(A) == len(B):
        x = len(A)
        C = np.concatenate((A, B), axis=1)
    else:
        print("Les matrices n'ont pas le même nombre de ligne")

    Mtrsup = ReductionGauss(C)
    X = ResolutionSystTriSup(Mtrsup)
    Result = np.asarray(X, dtype=float).reshape(x, 1)
    return Result


M = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
K = np.array([[7], [12], [3]])
F = np.array([[1, 1, 1, 1], [2, 4, -3, 2], [-1, -1, 0, -3], [1, -1, 4, 9]])
G = np.array([[1], [1], [2], [-8]])


# print(Gauss(M, K))


"""
DECOMPOSITION LU
"""

### QUESTION 1 ###

def DecompositionLU(A):
    """
    Rend la décomposition LU de la matrice A
    """
    U = A
    l = len(A)
    c = len(A[0])
    # print(l, c)
    L = np.eye(l) 
    # L = np.eye(l, l, dtype=int) 
    for i in range(0, l-1):
        # print(p)
        for k in range(i+1, l):
            g = U[k][i] / U[i][i]
            L[k][i] = g
            # print(g)
            for j in range(i, l):
                U[k][j] = U[k][j] - (g * U[i][j])
    
    return L, U

### QUESTION 2 ###

def ResolutionLU(L,B,U):
    """
    Résolution du sytème par la méthode LU
    """
    
    l = len(L)
    c = len(L[0])
    yinv = [0]*l    
    Linv = np.fliplr(np.flipud(L))
    Binv = np.flipud(B)
    # print(Linv)
    # print(Binv)
    x = [0]*l

    yinv[l-1] = Binv[l-1][0] / Linv[l-1][l-1]
    for i in range(l-2, -1, -1):
        yinv[i] = Binv[i][0]
        for j in range(i+1, l):
            yinv[i] = yinv[i] - Linv[i][j]*yinv[j]
        yinv[i] = (yinv[i]/Linv[i][i])
    y = np.flipud(yinv)

    x[l-1] = y[l-1] / U[l-1][l-1]
    for i in range(l-2, -1, -1):
        x[i] = y[i]
        for j in range(i+1, l):
            x[i] = x[i] - U[i][j]*x[j]
        x[i] = (x[i]/U[i][i])
    Result = np.asarray(x, dtype=float).reshape(len(L), 1)
    return Result

def LU(A, B):
    a = DecompositionLU(A)
    b = ResolutionLU(a[1], B, a[0])
    return b


# a = DecompositionLU(M)
# print("LU \n", ResolutionLU(a[1], K, a[0]))

"""
Pivot Partiel et Total
"""

def echange_ligne(A, i, j):
    """
    Echange les lignes d'indices i et j
    """
    A[i], A[j] = A[j], A[i]



def pivot_partiel_ligne(A, j0):
    """
    Donne l'indice du pivot de la colonne concernée
    """
    i = j0 #ligne du maximum provisoire
    modulepivot = abs(A[i][j0])
    for k in range(j0+1, len(A)):
        if abs(A[k][j0]) > modulepivot:
            i, modulepivot = k, abs(A[k][j0])
    return i


def copie(M):
    """
    Donne une copie de la matrice
    """
    l = len(M)
    c = len(M[0])
    M_copy = [[0]*c for _ in range(l)]
    for i in range(l):
        for j in range(c):
            M_copy[i][j]=M[i][j]
    return  M_copy

def mise_en_place_pivot_partiel(A, B, i):
    """
    Série d'opération pour appliquer le pivot partiel
    """
    j = pivot_partiel_ligne(A, i)

    if i != j:
        echange_ligne(A, i, j)
        echange_ligne(B, i, j)

def Gauss_pivot_partiel(A0, B0):
    """
    Résolution du système par la méthode de Gauss avec pivot partiel
    """

    A = copie(A0)
    B = copie(B0)
    l = len(A)

    for i in range(l-1):
        mise_en_place_pivot_partiel(A, B, i)

        for k in range(i+1, l):
            g = A[k][i] / A[i][i]
            # print(g)
            for j in range(i, l):
                A[k][j] = A[k][j] - (float(g) * A[i][j])
            B[k][0] = B[k][0] - (float(g) * B[i][0])
    
        C = np.concatenate((A, B), axis=1)
        c = len(C[0])
        x = [0]*l
        x[l-1] = C[l-1][c-1] / C[l-1][c-2]
    
        for i in range(l-2, -1, -1):
            x[i] = C[i][c-1]
            for j in range(i+1, l):
                x[i] = x[i] - C[i][j]*x[j]
            x[i] = (x[i]/C[i][i])

    Result = np.asarray(x, dtype=float).reshape(l, 1)
    return Result


def pivot_total_colonne(A, j0):
    """
    Donne les coordonnées du plus grand élément de la matrice A à chaque étape de la triangularisation
    """
    A = np.copy(A)
    i = j0
    if j0 != 0:
        M = np.copy(A)
        while i != 0:
            M = np.delete(M, 0, axis=0)
            M = np.delete(M, 0, axis=1)
            i -= 1
            # print(i, M)
        MaxCol = M.max(axis=0)
        PosMaxCol = MaxCol.argmax()
        PosMaxLig = M[:,PosMaxCol].argmax()
        # print(PosMaxCol)
        # print(PosMaxLig)
        a = [int(PosMaxLig)+j0, int(PosMaxCol)+j0]
    else:
        MaxCol = A.max(axis=0)
        PosMaxCol = MaxCol.argmax()
        PosMaxLig = A[:,PosMaxCol].argmax()
        # print(PosMaxCol)
        # print(PosMaxLig)
        a = [int(PosMaxLig), int(PosMaxCol)]
    return a 

def echange_colonne(A, i, j):
    """
    Echange les colonnes d'indice i et j
    """
    T = list()
    for o in range(len(A)):
        T.append(float(A[o][i]))
    for o in range(len(A)):
        A[o][i] = A[o][j]
        A[o][j] = T[o]

def echange_indice(l, i, j):
    """
    Echange la place des indice dans la liste quand on échange les colonnes d'indice i et j
    """
    T = l[i]
    l[i] = l[j]
    l[j] = T


def mise_en_place_pivot_total(A, B, l, i):
    """
    Série d'opération pour effectuer le pivot total
    """
    a = pivot_total_colonne(A, i)
    # print("l", a[0])
    # print("c", a[1])
    if i != a[1]:
        echange_colonne(A, i, a[1])
        echange_indice(l, i, a[1])
    if i != a[0]:
        echange_ligne(A, i, a[0])
        echange_ligne(B, i, a[0])
    
    j = pivot_partiel_ligne(A, i)
    if i != j:
        echange_ligne(A, i, j)
        echange_ligne(B, i, j)


def Gauss_pivot_total(A0, B0):
    """
    Résolution du système par la méthode de Gauss avec pivot total
    """
    
    A = copie(A0)
    B = copie(B0) 

    l = len(A)
    indice = [i for i in range(l)]

    for i in range(l-1):
        if i <= l-2:
            mise_en_place_pivot_total(A, B, indice, i)

        for k in range(i+1, l):
            g = A[k][i] / A[i][i]
            # print(g)
            for j in range(i, l):
                A[k][j] = A[k][j] - (float(g) * A[i][j])
            B[k][0] = B[k][0] - (float(g) * B[i][0])
    
        C = np.concatenate((A, B), axis=1)
        c = len(C[0])
        x = [0]*l
        x[l-1] = C[l-1][c-1] / C[l-1][c-2]
    
        for i in range(l-2, -1, -1):
            x[i] = C[i][c-1]
            for j in range(i+1, l):
                x[i] = x[i] - C[i][j]*x[j]
            x[i] = (x[i]/C[i][i])

    d_solution_desordre={}    # On remet les solutions dans l'ordre
    for i in range(l):
        d_solution_desordre[indice[i]] = x[i]
    d_solution_ordre = sorted(d_solution_desordre.items(), key=lambda t: t[0])
    
    solution = list()
    for i in range(l):
        solution.append(d_solution_ordre[i][1])

    Result = np.asarray(solution, dtype=float).reshape(l, 1)
    return Result


##### MATRICES TESTS #####

def matrice_alea():
    """
    Donne une liste contenant des listes de matrices aléatoires pour effectuer les tests
    """
    list_matrice = list()
    list_LU = list()
    list_test_LU = list()
    list_partiel = list()
    list_total = list()
    list_numpy = list()
    list_cond = list()
    for i in range(300, 600, 50):
        A = np.random.rand(i,i)
        B = np.copy(A)
        C = np.copy(A)
        D = np.copy(A)
        E = np.copy(A)
        F = np.copy(A)
        a = np.linalg.cond(A)
        list_cond.append(a)
        list_matrice.append(A)
        list_LU.append(B)
        list_test_LU.append(C)
        list_partiel.append(D)
        list_total.append(E)
        list_numpy.append(F)
    print("\nConditionnement moyen : ", st.mean(list_cond),"\n")
    return list_matrice, list_LU, list_test_LU, list_partiel, list_total, list_numpy



def vecteur_alea():
    """
    Donne une liste contenant les matrices colonnes aléatoires pour effectuer les tests
    """
    list_vecteur = list()
    for i in range(300, 600, 50):
        B = np.random.rand(i,1)
        list_vecteur.append(B)
    return list_vecteur

