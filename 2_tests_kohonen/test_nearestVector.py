#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@brief Script permettant de tester la fonction "nearestVector" de kohonen.py


"""




import kohonen
import matplotlib.pyplot as plt
import numpy


##test dans un cas favorable
input_vector = numpy.array([2,2])
vectors = numpy.array([[0,1],[3,3],[0,0],[1,1]])
d=kohonen.nearestVector(input_vector, vectors)

print(d)



##test dans un cas défavorable: tailles incompatibles
input_vector = numpy.array([2,2,2])
vectors = numpy.array([[0,1],[3,3],[0,0],[1,1]])
d=kohonen.nearestVector(input_vector, vectors)

print(d)