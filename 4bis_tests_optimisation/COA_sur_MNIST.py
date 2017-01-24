#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@brief Implémentation de l'algorithme de Kohonen et entraînement d'une Carte auto-organisatrice sur les imagettes.
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


def COA(data_path, iterations, nb_map, sigma_max_value, sigma_min_value, eta_max_value, eta_min_value, decay_start_iter,decay_stop_iter, affichage_graphique):
    """Génère une/des cartes COA

    @param data_path chemin d'accès aux données brutes
    @param iterations nombres d'itérations totales ayant servi à générer les COA
    @param nb_map nombre de fichiers COA 
    
    Paramètres de décroissance exponentielle (voir kohonen.py pour plus de précision)
    @param sigma_max_value (flottant) valeur du plateau initial pour sigma
    @param sigma_min_value (flottant) valeur du plateau final pour sigma
    @param eta_max_value (flottant) valeur du plateau initial pour eta 
    @param eta_min_value (flottant) valeur du plateau final pour eta
    @param decay_start_iter (entier) itération de démarrage de la décroissance exponentielle
    @param decay_stop_iter (entier) itération de fin de la décroissance exponentielle
    
    @return "weights/final_weights_%d".npy" carte COA contenue dans un fichier npy
    """    
    #===============================================================================
    # Chargement des données
    #===============================================================================
    file_handler = gzip.open(data_path, 'rb')
    data = pickle.load(file_handler, encoding='latin1')
    file_handler.close()
    
    #===============================================================================
    # Paramètres généraux de la simulation
    #===============================================================================
    
    ## affichage console d'information ou non
    verbose = False
    
    #===============================================================================
    # Récupération et Paramètres concernant les données d'apprentissage
    #===============================================================================
    
    training_data, training_label = data[0]
    labelling_data, labelling_labels = data[1]
    testing_data, testing_labels = data[2]
    
    ## dimension d'un vecteur d'entrée 
    data_shape = (28, 28)
    ## nombre de chiffres différents disponibles
    data_number = training_data.shape[0]
    ## dimensions des données : 'data_number' imagette de taille 'data_shape' générées aléatoirement
    data_dimension = (data_number, numpy.prod(data_shape))
    ## génération des données
    data = training_data
    
    #===============================================================================
    # Paramètres concernant la carte auto-organisatrice et l'algorithme de Kohonen
    #===============================================================================
    ## taille de la carte auto-organisatrice (COA)
    map_shape = (10, 10)
    ## dimensions des prototypes de la COA : une carte de MxM' [map_shape] vecteurs de dimension PxP' [data_dimension]
    weights_dimension = (numpy.prod(map_shape), numpy.prod(data_shape))
    ## initialisation aléatoire des prototypes de la COA : distribution uniforme entre 0. et 1.
    weights = numpy.random.random(size=weights_dimension)
    
    #===============================================================================
    # Boucle d'apprentissage suivant l'algorithme de Kohonen
    #===============================================================================
    # sampling_iter est le numéro des COA à prélever
    sampling_iter = iterations/nb_map
    
    for curr_iter in range(iterations):
        ## choisir un indice aléatoirement
        random_idx = numpy.random.randint(data_number)
        ## instancier l'exemple d'apprentissage courant
        sample = data[random_idx]
        ## récupérer les valeurs de sigma et eta
        sigma = kohonen.constrainedExponentialDecay(curr_iter, decay_start_iter, decay_stop_iter, sigma_max_value, sigma_min_value)
        eta = kohonen.constrainedExponentialDecay(curr_iter, decay_start_iter, decay_stop_iter, eta_max_value, eta_min_value)
        ## trouver la best-matching unit (BMU) et son score (plus petite distance)
        bmu_idx, bmu_score = kohonen.nearestVector(sample, weights)
        ## traduire la position 1D de la BMU en position 2D dans la carte
        bmu_2D_idx = numpy.unravel_index(bmu_idx, map_shape)
        ## gaussienne de taille sigma à la position 2D de la BMU
        gaussian_on_bmu = kohonen.twoDimensionGaussian(map_shape, bmu_2D_idx, sigma)
        ## mettre à jour les prototypes d'après l'algorithme de Kohonen (fonction à effets de bord)
        kohonen.updateKohonenWeights(sample, weights, eta, gaussian_on_bmu)
    
        ## afficher l'itération courante à l'écran
        if verbose: 
            print('Iteration %d/%d'%(curr_iter+1, iterations))
    
        if (curr_iter == sampling_iter):
            ## On sauve la COA ainsi obtenue
            numpy.save("weights/final_weights_%d"%(curr_iter),weights)
            sampling_iter += iterations/nb_map
    
    ## sauvegarde de la dernière COA
    numpy.save("weights/final_weights_%d"%(iterations),weights)
    
    
    
    if affichage_graphique:
        #===============================================================================
        # Affichage graphique
        #===============================================================================
        ## paramètrage d'un graphique affichant les prototypes d'imagettes appris par la COA
        ## création d'une nouvelle figure
        weights_plot = plt.figure('Imagettes associées aux prototypes de la carte')
        
        # parcours de la COA en ajoutant un subplot pour chaque neurone
        for image in range(map_shape[0]*map_shape[1]):
            ## création d'un axe matplotlib
            ax_weights = weights_plot.add_subplot(map_shape[0],map_shape[1],image)
            ## chargement dans la figure du neurone n°image comme matrice de pixels en niveau de gris
            ax_weights.imshow(weights[image,:].reshape(data_shape[0],data_shape[1]), interpolation='nearest', cmap = plt.cm.bone)
            ax_weights.axes.get_xaxis().set_visible(False)
            ax_weights.axes.get_yaxis().set_visible(False)
        
        ## affichage des graphiques
        plt.show()
