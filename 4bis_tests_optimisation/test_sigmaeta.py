#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@brief Script permettant de d'étudier et visualiser l'influence de sigma et eta sur le taux d'erreur. 
"""

#===============================================================================
# Importations nécessaires
#===============================================================================
import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D    # accès aux graphes 3D

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
nb_map = 1
iterations = 1000
    
## paramètres du rayon de voisinage gaussien (sigma)
sigma_max_value = 4.
sigma_min_value = .9
## paramètres du taux d'apprentissage (eta)
eta_max_value = .3
eta_min_value = .001
## paramètre de la décroissance exponentielle de sigma et eta
decay_start_iter = 0.2*iterations
decay_stop_iter = 0.6*iterations

LAB_all = False
DEC_all = False
affichage_graphique=False


#nombre de valeurs de sigma et eta à tester
sizeeta=2
sizesigma=2


# valeurs initiales de sigma et eta, pas de sigma et eta
sigma_ini = 0.5
sigma_step = 0.5

eta_ini = 0.04
eta_step= 0.04

#=============================================================================
# Fonction de visualisation
#=============================================================================

#fonction de visualisation en 3D de erreur=f(sigma,eta)
def scatterPlot(img):
    # tableau 2D a 3 colonnes pour les composantes sigma, eta, erreur
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(img[:,:,0].flat,    # tableau 1D des composantes rouges
               img[:,:,1].flat,    # tableau 1D des composantes vertes
               img[:,:,2].flat,    # tableau 1D des composantes bleues
               )
    plt.show()


#=============================================================================
# Code
#=============================================================================

# array contenant le taux d'erreur en fonction de sigma et eta
erreur_f_sigma_eta = numpy.zeros((sizesigma,sizeeta,3))
i=0
j=0
curr_iter = 0

for i in range(sizesigma):
    sigma_max_value = sigma_ini + i*sigma_step
    for j in range(sizeeta):
        eta_max_value = eta_ini + j*eta_step
        COA_sur_MNIST.COA(data_path, iterations, nb_map, sigma_max_value, sigma_min_value, eta_max_value, eta_min_value, decay_start_iter,decay_stop_iter, affichage_graphique)
        LAB_sur_MNIST.LAB(data_path, iterations, nb_map, LAB_all, None)
        error_rate = DECISION.DEC(data_path, iterations, nb_map, LAB_all, DEC_all, None)
        
        erreur_f_sigma_eta[i,j,0] = sigma_max_value
        erreur_f_sigma_eta[i,j,1] = eta_max_value
        erreur_f_sigma_eta[i,j,2] = error_rate
        
        curr_iter = curr_iter+1
        print(curr_iter)
        
        
scatterPlot(erreur_f_sigma_eta)
        

