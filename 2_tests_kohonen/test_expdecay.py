#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@brief Script permettant de tester la fonction "constrainedExponentialDecay" de
kohonen.py


"""




import kohonen
import matplotlib.pyplot as plt
import numpy


## test dans un cas favorable
plot=numpy.empty(100)

for cur_iter in range(100):
    plot[cur_iter]=kohonen.constrainedExponentialDecay(cur_iter,60,60,50,10)
    
plt.plot(plot)
plt.show()

##test d'erreur: si start_iter>stop_iter
plot=numpy.empty(100)

for cur_iter in range(100):
    plot[cur_iter]=kohonen.constrainedExponentialDecay(cur_iter,80,60,50,10)
    
plt.plot(plot)
plt.show()

##test d'erreur: si max_value<min_value
plot=numpy.empty(100)

for cur_iter in range(100):
    plot[cur_iter]=kohonen.constrainedExponentialDecay(cur_iter,200,60,10,50)
    
plt.plot(plot)
plt.show()