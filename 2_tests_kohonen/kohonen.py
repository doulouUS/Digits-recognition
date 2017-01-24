#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@brief Ensemble des fonctions essentielles à l'implémentation en Python de l'algorithme de Kohonen en utilisant NumPy

--> Version avec NumPy
--> Version avec décroissance exponentielle pour sigma et eta

Code implémenté en vue du projet Ensta IN104-Python 2014-2015

@author Thomas Hecht
@version 0.2.9
"""

#===============================================================================
# Importation des modules nécessaires
#===============================================================================
import numpy

#===============================================================================
# Déclaration des fonctions
#===============================================================================

def constrainedExponentialDecay(curr_iter, start_iter, stop_iter, max_value, min_value):
    """Renvoie la valeur d'une décroissance exponentielle entre bornes : avant start_iter, max_value ; après stop_iter, min_value ; entre les deux, on passe exponentiellement de max_value à min_value

    @param curr_iter (entier) itération courante
    @param start_iter (entier) itération de démarrage de la décroissance exponentielle
    @param stop_iter (entier) itération de fin de la décroissance exponentielle   
    @param max_value (flottant) valeur du plateau initial    
    @param min_value (flottant) valeur du plateau final    
    @see http://fr.wikipedia.org/wiki/D%C3%A9croissance_exponentielle
    @return (flottant) valeur de la décroissance exponentielle bornée
    """
    #si start_iter>stop iter, incompatibilité
    if start_iter>stop_iter:
        print("Incompatible, start_iter doit etre inférieur à stop_iter")
        return None
        
    #si max_value<min_value, incompatibilité
    if max_value<min_value:
        print("incompatible, max_value doit etre superieur a min_value")
        return None
    
    #avant start_iter : on renvoie max_value
    if curr_iter <= start_iter :
        return max_value
    #après stop_iter : on renvoie min_value       
    if curr_iter > stop_iter :
        return min_value        
    #entre start_iter et stop_iter : on calcule l'exponentielle décroissante         
    k = numpy.log(min_value/max_value)/(start_iter-stop_iter)
    alpha = min_value / numpy.exp(-k*stop_iter)
    return alpha * numpy.exp(-curr_iter*k)

def nearestVector(input_vector, vectors):
    """Renvoie l'indice (et la distance associée) du vecteur (issu de vectors) "le plus proche" du vecteur input_vector au sens de la distance euclidienne

    @param input_vector (numpy.ndarray) vecteur d'entrée unique de dimension n
    @param vectors (numpy.ndarray) vecteur de vecteurs de dimension m * n
    @return (entier, flottant) indice et distance du vecteur le plus proche
    """
    #calcul de la distance euclidienne entre chaque vecteur de 'vectors' et 'input_vector'
    distances_matrix = numpy.sqrt(numpy.sum((vectors-input_vector)**2, axis=1))
    #récupération de la BMU et de son "score"
    index_of_nearest = numpy.argmin(distances_matrix)   
    score_of_nearest = numpy.min(distances_matrix)
    return index_of_nearest, score_of_nearest

def twoDimensionGaussian(space_shape, gaussian_position, gaussian_sigma):
    """Renvoie un noyau gaussien à une certain position sur une grille en 2D avec une variance de gaussian_variance de maximum valant 1.0.

    @param space_shape (tuple) taille de l'espace 2D au format (m, n)
    @param gaussian_position (tuple) position du pic de la gaussienne au format (y, x)
    @param gaussian_sigma (flottant) écart-type de la gaussienne, en termes de l'espace 2D (c-à-d en pixels si l'on assimile espace 2D à une image)
    @see http://fr.wikipedia.org/wiki/Fonction_gaussienne
    @see http://docs.scipy.org/doc/numpy/reference/generated/numpy.ogrid.html
    @return (numpy.ndarray) vecteur de vecteurs représentant une 'bulle' gaussienne (à aplatir avant de renvoyer)
    """
    #création de deux rampes "ouvertes" (en opposition avec la caractère "dense" des mgrid) d'indices
    M, N = numpy.ogrid[0:space_shape[0], 0:space_shape[1]]
    #projection de la position du pic de la gaussienne
    Y, X = M - gaussian_position[0], N - gaussian_position[1]
    #calcul du noyau gaussien centré
    kernel = numpy.exp(-(X ** 2 + Y ** 2) / (2 * gaussian_sigma ** 2))
    return kernel.ravel()

def updateKohonenWeights(input_vector, weights, learning_rate, neighborhood):
    """Renvoie les prototypes mis à jour d'après l'algorithme de Kohonen. - Effets de bord

    @param input_vector (numpy.ndarray) vecteur d'entrée unique de dimension n
    @param weights (numpy.ndarray) vecteur des poids de dimension m * n
    @param learning_rate (flottant) taux d'apprentissage [eta]
    @param neighborhood (numpy.ndarray) gaussienne sur un espace 2D ('aplatie' à m * n) intégrant implicitement la fonction de voisinage
    @see twoDimensionGaussian()
    @see nearestVector()
    @return None
    """
    #mise à jour (avec effets de bord) des poids du voisinage de la BMU avec un pourcentage de la distance entre BMU et exemple courant
    weights += learning_rate * (input_vector[numpy.newaxis, :] - weights) * neighborhood.ravel()[:, numpy.newaxis]
    
