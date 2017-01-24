#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@brief Implémentation de l'algorithme de labellisation sur les cartes auto-organisatrices à partir d'imagettes de référence
"""

#===============================================================================
# Importations nécessaires
#===============================================================================
import matplotlib.pyplot as plt
import numpy

import kohonen
import pickle #gérer parti .pkl (parti sérialisé)
import gzip #gérer parti .gz (compression)


def LAB(data_path, iterations, nb_map, LAB_all, LAB_nb):
    """Génère une/des cartes labellisée à partir de fichier COA

    @param data_path chemin d'accès aux données brutes
    @param iterations nombres d'itérations totales ayant servi à générer les COA
    @param nb_map nombre de fichiers COA 
    @param LAB_all booléen permettant de choisir si l'on veut labelliser toutes les
           COA présentes ou bien une seule
    @param LAB_nb numéro du fichier COA qu'on veut labelliser ou None
    @return "labelled_map/label_map_%d.npy" carte labellisée contenue dans un ficher npy
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
    verbose = True
    
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
    
    
    #===============================================================================
    # LABELLISATION
    #===============================================================================
    
    # Cas ou l'on souhaite labelliser toutes les cartes
    if LAB_all == True:  
    	for weights_nb in range(1,nb_map+1):
            #==========
            # Chargement de la COA
            #==========
        	weights=numpy.load("weights/final_weights_%d.npy"%(weights_nb*iterations/nb_map))
            ...
        
            #===============================================================================
            # Labellisation
            #===============================================================================
        
            data = labelling_data
            sigma_LAB = 0.5
        
            ## création d'un tableau de même taille que la COA et contenant un tableau de score pour les chiffres de 0 à 9
            label_scores_dimension = (numpy.prod(map_shape), 10)
            label_scores = numpy.zeros(label_scores_dimension)
        
            for curr_iter in range(data.shape[0]):
                ## instancier l'exemple de labellisation courant
                sample = data[curr_iter]
                ## trouver la best-matching unit (BMU) et son score (plus petite distance)
                bmu_idx, bmu_score = kohonen.nearestVector(sample, weights)
                ## traduire la position 1D de la BMU en position 2D dans la carte
                bmu_2D_idx = numpy.unravel_index(bmu_idx, map_shape)
        
                ## mettre à jour les scores du tableau (à implémenter une influence sur les voisins)
                label_scores[:, labelling_labels[curr_iter]] += kohonen.twoDimensionGaussian(map_shape, bmu_2D_idx, sigma_LAB)
                ## afficher l'itération courante à l'écran
                if verbose: 
                    print('Iteration %d/%d'%(curr_iter+1, data.shape[0]))
        
            ## attribution du chiffre correspondant au score maximal
            label_scores = numpy.argmax(label_scores, axis=1)
        
        
            ## On sauve la carte labellisée
            numpy.save("labelled_map/label_map_%d"%(int((weights_nb*iterations)/nb_map)), label_scores)
            
    else:
    # Cas où l'on ne souhaite labelliser qu'un seule carte
        ## Par défaut, si aucun nom n'est précisé, labellisée la COA la plus aboutie
        if LAB_nb == None:
            #======================
            # Chargement de la COA
            #======================
        
            weights=numpy.load("weights/final_weights_%d.npy"%(iterations))
            
            ...
        
            #===============================================================================
            # Labellisation
            #===============================================================================
        
            data = labelling_data
            sigma_LAB = 0.5
        
            ## création d'un tableau de même taille que la COA et contenant un tableau de score pour les chiffres de 0 à 9
            label_scores_dimension = (numpy.prod(map_shape), 10)
            label_scores = numpy.zeros(label_scores_dimension)
        
            for curr_iter in range(data.shape[0]):
                ## instancier l'exemple de labellisation courant
                sample = data[curr_iter]
                ## trouver la best-matching unit (BMU) et son score (plus petite distance)
                bmu_idx, bmu_score = kohonen.nearestVector(sample, weights)
                ## traduire la position 1D de la BMU en position 2D dans la carte
                bmu_2D_idx = numpy.unravel_index(bmu_idx, map_shape)
        
                ## mettre à jour les scores du tableau (à implémenter une influence sur les voisins)
                label_scores[:, labelling_labels[curr_iter]] += kohonen.twoDimensionGaussian(map_shape, bmu_2D_idx, sigma_LAB)
                ## afficher l'itération courante à l'écran
                if verbose: 
                    print('Iteration %d/%d'%(curr_iter+1, data.shape[0]))
        
            ## attribution du chiffre correspondant au score maximal
            label_scores = numpy.argmax(label_scores, axis=1)
        
        
            ## On sauve la carte labellisée
            numpy.save("labelled_map/label_map_%d"%(iterations), label_scores)
            
            
        ## si un numéro de COA est précisé, charger ce fichier
        else:
            #===============================================================================
            # Chargement de la COA
            #===============================================================================
        
            weights=numpy.load(LAB_nb)
        
            #===============================================================================
            # Labellisation
            #===============================================================================
        
            data = labelling_data
            sigma_LAB = 0.5
        
            ## création d'un tableau de même taille que la COA et contenant un tableau de score pour les chiffres de 0 à 9
            label_scores_dimension = (numpy.prod(map_shape), 10)
            label_scores = numpy.zeros(label_scores_dimension)
        
            for curr_iter in range(data.shape[0]):
                ## instancier l'exemple de labellisation courant
                sample = data[curr_iter]
                ## trouver la best-matching unit (BMU) et son score (plus petite distance)
                bmu_idx, bmu_score = kohonen.nearestVector(sample, weights)
                ## traduire la position 1D de la BMU en position 2D dans la carte
                bmu_2D_idx = numpy.unravel_index(bmu_idx, map_shape)
        
                ## mettre à jour les scores du tableau (à implémenter une influence sur les voisins)
                label_scores[:, labelling_labels[curr_iter]] += kohonen.twoDimensionGaussian(map_shape, bmu_2D_idx, sigma_LAB)
                ## afficher l'itération courante à l'écran
                if verbose: 
                    print('Iteration %d/%d'%(curr_iter+1, data.shape[0]))
        
            ## attribution du chiffre correspondant au score maximal
            label_scores = numpy.argmax(label_scores, axis=1)
        
        
            ## On sauve la carte labellisée
            numpy.save("labelled_map/label_map_%d"%(LAB_nb), label_scores)
    
    
    ## On réordonne label_scores pour le printer dans les dimensions map_shape
    labelled_card = numpy.reshape(label_scores,map_shape)
    print(labelled_card)
    
    
