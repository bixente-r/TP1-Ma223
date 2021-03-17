import TP1_lib as tp
import numpy as np
import time
import math as m
import statistics as st
import matplotlib.pyplot as plt

M = tp.matrice_alea()
V = tp.vecteur_alea()
F = np.array([[1, 1.01, 1, 1], [2, 4, -3, 2], [-1, -1, 0, -3], [1, -1, 4, 9]])
G = np.array([[1], [1], [2], [-8]])

debut = 300 # tailles des  matrices pour la création des graphiques
fin = 600
pas = 50

##### LISTES DES TEMPS ET DIMENSIONS GAUSS #####

l_temps = list()
l_log_temps = list()

l_n = [j for j in range(debut, fin, pas)]

l_log_n = [m.log(j) for j in l_n]


l_resultats = list()
l_erreur = list()


##### BOUCLE PRINCIPALE GAUSS #####
ti = time.time()
for i in range(len(M[0])):
    temps_init = time.time()
    l_resultats.append(tp.Gauss(M[0][i], V[i])) # on applique la méthode et on mesure le temps
    temps_fin = time.time()

    temps_calcul = temps_fin - temps_init
    l_temps.append(temps_calcul)
    if l_temps[i] != 0:
        l_log_temps.append(m.log(l_temps[i]))
    else:
        l_log_temps.append(-30)
    err = np.linalg.norm(M[0][i].dot(l_resultats[i])-V[i]) # on calcule la norme de l'erreur ||AX-B||
    l_erreur.append(err)

l_erreur_log = [m.log(j) for j in l_erreur]

modele_gauss=np.polyfit(l_log_n, l_log_temps, 1) # on crée une courbe de tendance polynomiale d'ordre 1 (une droite)
equation_gauss=("Gauss, y = "+format(modele_gauss[0],".2e")+"x + "+format(modele_gauss[1],".2e"))

print("\n###### Méthode de Gauss sans pivot ######")
print("Erreur moyenne : ", st.mean(l_erreur))
tf = time.time()
temps_calcul_total = time.strftime('%H hrs %M min %S sec', time.gmtime(tf-ti))
print("Temps total de calcul :", temps_calcul_total)
print(l_erreur)

##### LISTES DES TEMPS ET DIMENSIONS LU #####

l_temps_LU = list()
l_log_temps_LU = list()

l_n_LU = [j for j in range(debut, fin, pas)]

l_log_n_LU = [m.log(j) for j in l_n_LU]


l_resultats_LU = list()
l_erreur_LU = list()


##### BOUCLE PRINCIPALE LU #####

ti = time.time()
for i in range(len(M[0])):
    
    temps_init = time.time()
    lu = tp.DecompositionLU(M[1][i])
    l_resultats_LU.append(tp.ResolutionLU(lu[0], V[i], lu[1]))
    temps_fin = time.time()

    temps_calcul = temps_fin - temps_init
    l_temps_LU.append(temps_calcul)
    if l_temps_LU[i] != 0:
        l_log_temps_LU.append(m.log(l_temps_LU[i]))
    else:
        l_log_temps_LU.append(-30)

    err = np.linalg.norm(np.dot(M[2][i], l_resultats_LU[i])-V[i])
    l_erreur_LU.append(err)

l_erreur_LU_log = [m.log(j) for j in l_erreur_LU]


modele_LU = np.polyfit(l_log_n_LU, l_log_temps_LU, 1)
equation_LU = ("LU, y = "+format(modele_LU[0],".2e")+"x + "+format(modele_LU[1],".2e"))

print("\n###### Méthode LU ######")
print("Erreur moyenne : ", st.mean(l_erreur_LU))
tf = time.time()
temps_calcul_total = time.strftime('%H hrs %M min %S sec', time.gmtime(tf-ti))
print("Temps total de calcul :", temps_calcul_total)
print(l_erreur_LU)


##### LISTES DES TEMPS ET DIMENSIONS GAUSS partiel #####

l_temps_partiel = list()
l_log_temps_partiel = list()

l_n_partiel = [j for j in range(debut, fin, pas)]
# l_n_erreur_partiel = [j for j in range(100, 800, 10)]

l_log_n_partiel = [m.log(j) for j in l_n_partiel]


l_resultats_partiel = list()
l_erreur_partiel = list()
index = list()

##### BOUCLE PRINCIPALE GAUSS partiel #####
ti = time.time()
for i in range(len(M[0])):
    temps_init = time.time()
    l_resultats_partiel.append(tp.Gauss_pivot_partiel(M[3][i], V[i]))
    temps_fin = time.time()

    temps_calcul = temps_fin - temps_init
    l_temps_partiel.append(temps_calcul)
    if l_temps_partiel[i] != 0:
        l_log_temps_partiel.append(m.log(l_temps_partiel[i]))
    else:
        l_log_temps_partiel.append(-30)
    err = np.linalg.norm(np.dot(M[3][i], l_resultats_partiel[i])-V[i])
    l_erreur_partiel.append(err)

"""    if err > 1*10**-6:
        index.append(l_n_erreur_partiel[i])
    else:
        l_erreur_partiel.append(err)"""

"""if len(index) != 0:
    l_n_erreur_partiel = [e for e in l_n_erreur_partiel if e not in index]
else:
    pass"""

l_erreur_partiel_log = [m.log(j) for j in l_erreur_partiel]

modele_partiel = np.polyfit(l_log_n_partiel, l_log_temps_partiel, 1)
equation_partiel = ("Gauss Partiel, y = "+format(modele_partiel[0],".2e")+"x + "+format(modele_partiel[1],".2e"))

print("\n###### Méthode de Gauss pivot partiel ######")
print("Erreur moyenne : ", st.mean(l_erreur_partiel))
tf = time.time()
temps_calcul_total = time.strftime('%H hrs %M min %S sec', time.gmtime(tf-ti))
print("Temps total de calcul :", temps_calcul_total)
print(l_erreur_partiel)

##### LISTES DES TEMPS ET DIMENSIONS GAUSS total #####

l_temps_total = list()
l_log_temps_total = list()

l_n_total = [j for j in range(debut, fin, pas)]
l_n_erreur_total = [j for j in range(debut, fin, pas)]

l_log_n_total = [m.log(j) for j in l_n_total]


l_resultats_total = list()
l_erreur_total = list()


##### BOUCLE PRINCIPALE GAUSS total #####
ti = time.time()
for i in range(len(M[0])):
    temps_init = time.time()
    l_resultats_total.append(tp.Gauss_pivot_total(M[4][i], V[i]))
    temps_fin = time.time()

    temps_calcul = temps_fin - temps_init
    l_temps_total.append(temps_calcul)
    if l_temps_total[i] != 0:
        l_log_temps_total.append(m.log(l_temps_total[i]))
    else:
        l_log_temps_total.append(-30)
    err = np.linalg.norm(np.dot(M[4][i], l_resultats_total[i])-V[i])
    l_erreur_total.append(err)

l_erreur_total_log = [m.log(j) for j in l_erreur_total]

modele_total=np.polyfit(l_log_n_total, l_log_temps_total, 1)
equation_total=("Gauss Total, y = "+format(modele_total[0],".2e")+"x + "+format(modele_total[1],".2e"))

print("\n###### Méthode de Gauss pivot total ######")
print("Erreur moyenne : ", st.mean(l_erreur_total))
tf = time.time()
temps_calcul_total = time.strftime('%H hrs %M min %S sec', time.gmtime(tf-ti))
print("Temps total de calcul :", temps_calcul_total)
print(l_erreur_total)



##### LISTES DES TEMPS ET DIMENSIONS numpy #####

l_temps_numpy = list()
l_log_temps_numpy = list()

l_n_numpy = [j for j in range(debut, fin, pas)]

l_log_n_numpy = [m.log(j) for j in l_n_numpy]

l_resultats_numpy = list()
l_erreur_numpy = list()


##### BOUCLE PRINCIPALE numpy #####
ti = time.time()
for i in range(len(M[0])):
    temps_init = time.time()
    l_resultats_numpy.append(np.linalg.solve(M[5][i], V[i]))
    temps_fin = time.time()

    temps_calcul = temps_fin - temps_init
    l_temps_numpy.append(temps_calcul)
    if l_temps_numpy[i] != 0:
        l_log_temps_numpy.append(m.log(l_temps_numpy[i]))
    else:
        l_log_temps_numpy.append(-30)
    err = np.linalg.norm(np.dot(M[4][i], l_resultats_numpy[i])-V[i])
    l_erreur_numpy.append(err)

l_erreur_numpy_log = [m.log(j) for j in l_erreur_numpy]

modele_numpy = np.polyfit(l_log_n_numpy, l_log_temps_numpy, 1)
equation_numpy = ("Numpy, y = "+format(modele_numpy[0],".2e")+"x + "+format(modele_numpy[1],".2e"))
print(equation_numpy)


print("\n###### Solveur Numpy ######")
print("Erreur moyenne : ", st.mean(l_erreur_numpy))
tf = time.time()
temps_calcul_total = time.strftime('%H hrs %M min %S sec', time.gmtime(tf-ti))
print("Temps total de calcul :", temps_calcul_total)
print(l_erreur_numpy)




plt.title("Temps de résolution en fonction de n")
plt.plot(l_n, l_temps, "o-", color="green", label="Gauss")
plt.plot(l_n_LU, l_temps_LU, "o--", color="orange", label="LU")
plt.plot(l_n_partiel, l_temps_partiel, "o:", color="blue", label="Gauss Partiel")
plt.plot(l_n_total, l_temps_total,"o-.", color="magenta", label="Gauss Total")
plt.plot(l_n_numpy, l_temps_numpy, "o-", color="cyan", label="Numpy solve()")
plt.xlabel("Dimension, n")
plt.ylabel("temps de calcul (sec)")
plt.legend()
plt.show()

plt.title("[log-log] Temps de résolution en fonction de n")
plt.plot(l_log_n, l_log_temps, "o-", color="green", label=equation_gauss)
plt.plot(l_log_n_LU, l_log_temps_LU, "o--", color="orange", label=equation_LU)
plt.plot(l_log_n_partiel, l_log_temps_partiel, "o:", color="blue", label=equation_partiel)
plt.plot(l_log_n_total, l_log_temps_total,"o-.", color="magenta", label=equation_total)
plt.plot(l_log_n_numpy, l_log_temps_numpy, "o-", color="cyan", label="Numpy solve()")#label=equation_numpy
plt.xlabel("Log Dimension, log(n)")
plt.ylabel("Log temps de calcul (sec)")
plt.legend()
plt.show()

plt.title("Graphe des erreurs en fonction de n")
plt.plot(l_n, l_erreur, "o-", color="green", label="Gauss")
plt.plot(l_n_LU, l_erreur_LU, "o--", color="orange", label="LU")
plt.plot(l_n_partiel, l_erreur_partiel, "o:", color="blue", label="Gauss Partiel")
plt.plot(l_n_total, l_erreur_total,"o-.", color="magenta", label="Gauss Total")
plt.plot(l_n_numpy, l_erreur_numpy, "o-", color="cyan", label="Numpy solve()")
plt.xlabel("Dimension, n")
plt.ylabel("Erreur ||Ax-B||")
plt.legend()
plt.show()

plt.title("Log Erreurs")
plt.plot(l_n, l_erreur_log, "o-", color="green", label="Gauss")
plt.plot(l_n_LU, l_erreur_LU_log, "o--", color="orange", label="LU")
plt.plot(l_n_partiel, l_erreur_partiel_log, "o:", color="blue", label="Gauss Partiel")
plt.plot(l_n_total, l_erreur_total_log,"o-.", color="magenta", label="Gauss Total")
plt.plot(l_n_numpy, l_erreur_numpy_log, "o-", color="cyan", label="Numpy solve()")
plt.xlabel("Dimension, n")
plt.ylabel("Erreur Log(||Ax-B||)")
plt.legend()
plt.show()