# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 18:38:55 2021

@author: Jackie Smale

Make sure our functions work! 
"""
import numpy as np
import matplotlib.pyplot as plt
import numericalFunctions as nf

# Use (a) fixed-point iteration and (b) Newton-Raphson method to determine
# a root of f(x) = -0.9*x^2 +1.7*x + 2.5 using x_0 = 5.  Perform the
# computation until e_a is less than e_s = 0.001%.

# Graphical root estimate
# not required in question, but nice to see what the eqn looks like! 
xGraph = np.arange(-10, 10.01, 0.01)
y =  - 0.9*xGraph**2 + 1.7*xGraph + 2.5;
null = np.zeros(len(xGraph)) #zeros so finding the root is easier

#Plot Graphical Estimate 
fig = plt.figure()
plt.plot(xGraph,y)
plt.plot(xGraph, null, 'r--')
plt.title('Estimate root graphically')


# A) Fixed- point iteration method
# Making sure we develop a soln that converges on the root

x0 = 5 #intial guess

# Rearrange function so that x = g(x)
 # x = ((1.7x+2.5)/0.9)^0.5
g = lambda x: np.sqrt((1.7*x + 2.5)/0.9)

[xrB, eps_aB, iterB] = nf.fixed_point(x0, g)

# B) Newton Raphson Method 
#f to pass into function 
f = lambda x:  -0.9*x**2 + 1.7*x + 2.5
#derivative wrt x to pass into root_newton_raphson
dfdx = lambda x: -1.8*x - 1.7;

[xrC,eps_aC, iterC] = nf.root_newton_raphson(x0,f,dfdx)