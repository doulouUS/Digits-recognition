#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@brief Script permettant de d'étudier et visualiser l'influence du nombre d'itération sur le taux d'erreur. 
On désire connaître le taux d'erreur en fonction du nombre d'itérations: on va prélever une version de la COA
en cours de réalisation à une certaine fréquence
Cela permet de déterminer le nombre d'itérations à partir duquel il est inutile de laisser tourner le code.
"""

#===============================================================================
# Importations nécessaires
#===============================================================================
import matplotlib.pyplot as plt
import numpy
#création de dossier par python
import os

import kohonen
import pickle #gérer parti .pkl (parti sérialisé)
import gzip #gérer parti .gz (compression)

import COA_sur_MNIST
import LAB_sur_MNIST
import DECISION


#===============================================================================
# Paramètres concernant les données			#** à saisir **#
#===============================================================================
## chemin d'accès local au fichier mnist.pkl.gz

data_path = "/home/b/beng/IN104/data/mnist.pkl.gz"	


#===============================================================================
# Paramètres généraux de la simulation			#** à saisir **#
#===============================================================================

## nombre total d'itérations d'apprentissage demandée à l'utilisateur
nb_map = 4
iterations = 10000
    
## paramètres du rayon de voisinage gaussien (sigma)
sigma_max_value = 2
sigma_min_value = .9
## paramètres du taux d'apprentissage (eta)
eta_max_value = .2
eta_min_value = .001
## paramètre de la décroissance exponentielle de sigma et eta
decay_start_iter = 0.2*iterations
decay_stop_iter = 0.6*iterations

LAB_all = True
DEC_all = True
affichage_graphique=True




#===============================================================================
# Code
#===============================================================================

COA_sur_MNIST.COA(data_path, iterations, nb_map, sigma_max_value, sigma_min_value, eta_max_value, eta_min_value, decay_start_iter,decay_stop_iter, affichage_graphique)
LAB_sur_MNIST.LAB(data_path, iterations, nb_map, LAB_all, None)
error_rate = DECISION.DEC(data_path, iterations, nb_map, LAB_all, DEC_all, None)

x = error_rate[:,0]
y = error_rate[:,1]
plt.plot(x,y)
plt.show()
