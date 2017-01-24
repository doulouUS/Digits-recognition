#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@brief Implémentation de l'algorithme de décision: sur un ensemble d'images 
n'ayant servi ni à entrainer ni à labelliser, le script prend des décision 
concernant leur classe et un taux d'erreur sera généré
"""

#===============================================================================
# Importations nécessaires
#===============================================================================
import matplotlib.pyplot as plt
import numpy

import kohonen
import pickle #gérer parti .pkl (parti sérialisé)
import gzip #gérer parti .gz (compression)


def DEC(data_path, iterations, nb_map, LAB_all, DEC_all, DEC_nb):
    """Génère un taux d'erreur sur cartes labellisées

    @param data_path chemin d'accès aux données brutes
    @param iterations nombres d'itérations totales ayant servi à générer les COA
    @param nb_map nombre de fichiers COA 
    @param DEC_all booléen permettant de choisir si l'on veut évaluer tous les
           couples (COA, cartes labellisées) présents ou bien un seul
    @param DEC_nb numéro du couple qu'on veut labelliser ou None
    @return taux d'erreur sous forme d'un numpy array si DEC_all=True, sinon sous forme d'un float
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
        
    
    # Cas où l'on souhaite labelliser toutes les cartes
    if DEC_all == True:  
        if LAB_all == False:
            print("incompatibilité: il n'y a pas assez de cartes labellisées")
            return(None)
        
        
        #tableau contenant le taux d'erreur en fonction de la carte analysée
        error_rate = numpy.zeros((nb_map,2))
        k=0
        
        for weights_nb in range(1,nb_map+1):
            #===============================================================================
            # Chargement de la carte entraînée
            #===============================================================================
            
            label_scores = numpy.load("labelled_map/label_map_%d.npy"%((weights_nb*iterations)/nb_map))
            weights=numpy.load("weights/final_weights_%d.npy"%((weights_nb*iterations)/nb_map))
            
            
            #===============================================================================
            # Prise de décision
            #===============================================================================
            
            data = testing_data
            
            ## création d'un tableau de même taille que la COA et contenant un tableau de score
            test_scores_dimension = (numpy.prod(map_shape), 1)
            test_scores = numpy.zeros(test_scores_dimension)
            
            for curr_iter in range(data.shape[0]):
                ## instancier l'exemple de labellisation courant
                sample = data[curr_iter]
                ## trouver la best-matching unit (BMU) et son score (plus petite distance)
                bmu_idx, bmu_score = kohonen.nearestVector(sample, weights)
                ## mettre à jour les scores du tableau 
                if testing_labels[curr_iter] != label_scores[bmu_idx]:
                    test_scores[bmu_idx] += 1
                ## afficher l'itération courante à l'écran
                if verbose: 
                    print('Iteration %d/%d'%(curr_iter+1, data.shape[0]))
            
            error_rate[k,0] = (weights_nb*iterations)/nb_map
            error_rate[k,1] = numpy.sum(test_scores)/100
            
            k=k+1
            
        return(error_rate)
        
    else:
    # Cas où l'on ne souhaite labelliser qu'un seule carte
        ## Par défaut, si aucun nom n'est précisé, labellisée la COA la plus aboutie
        if DEC_nb == None:
            #===============================================================================
            # Chargement de la carte entraînée
            #===============================================================================
            
            label_scores = numpy.load("labelled_map/label_map_%d.npy"%(iterations))
            weights=numpy.load("weights/final_weights_%d.npy"%(iterations))
            
            
            #===============================================================================
            # Prise de décision
            #===============================================================================
            
            data = testing_data
            
            ## création d'un tableau de même taille que la COA et contenant un tableau de score
            test_scores_dimension = (numpy.prod(map_shape), 1)
            test_scores = numpy.zeros(test_scores_dimension)
            
            for curr_iter in range(data.shape[0]):
                ## instancier l'exemple de labellisation courant
                sample = data[curr_iter]
                ## trouver la best-matching unit (BMU) et son score (plus petite distance)
                bmu_idx, bmu_score = kohonen.nearestVector(sample, weights)
                ## mettre à jour les scores du tableau 
                if testing_labels[curr_iter] != label_scores[bmu_idx]:
                    test_scores[bmu_idx] += 1
                ## afficher l'itération courante à l'écran
                if verbose: 
                    print('Iteration %d/%d'%(curr_iter+1, data.shape[0]))
            
            
            return(numpy.sum(test_scores)/100)
        
        #si un numéro de carte est précisé, évaluer celle là
        else:
            #===============================================================================
            # Chargement de la carte entraînée
            #===============================================================================
            
            label_scores = numpy.load("labelled_map/label_map_%d.npy"%(DEC_nb))
            weights=numpy.load("weights/final_weights_%d.npy"%(DEC_nb))
            
            
            #===============================================================================
            # Prise de décision
            #===============================================================================
            
            data = testing_data
            
            ## création d'un tableau de même taille que la COA et contenant un tableau de score
            test_scores_dimension = (numpy.prod(map_shape), 1)
            test_scores = numpy.zeros(test_scores_dimension)
            
            for curr_iter in range(data.shape[0]):
                ## instancier l'exemple de labellisation courant
                sample = data[curr_iter]
                ## trouver la best-matching unit (BMU) et son score (plus petite distance)
                bmu_idx, bmu_score = kohonen.nearestVector(sample, weights)
                ## mettre à jour les scores du tableau 
                if testing_labels[curr_iter] != label_scores[bmu_idx]:
                    test_scores[bmu_idx] += 1
                ## afficher l'itération courante à l'écran
                if verbose: 
                    print('Iteration %d/%d'%(curr_iter+1, data.shape[0]))
            
            
            return(numpy.sum(test_scores)/100)
            
            
            

    

