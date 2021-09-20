# -*- coding: utf-8 -*-
"""
Jackie Smale

9/19/2021
"""
import numpy as np
import matplotlib.pyplot as plt

def root_newton_raphson(x0, f, dfdx):
    """   
     Newton Rapson method 
    Function takes in three values 
      x0 = intital guess
      f = functiona handle of desired function 
     dfdx = a function that gives the first derivative
    Outputs 3 values
      xr = estimated root
      iter = how many iterations the algorithm took 
      epsA = vector containting the error
    """
    epsA = []; epsA.append(1); 
    epsS = 1e-5; 
    i = 0 #iteration counter
    maxiter = 10000 #maximum number of iterations
    xr = x0 #Set estimated root to initial guess
    
    while epsA[-1] > epsS and i < maxiter:
            fxr = f(xr)
            dfdxr = dfdx(xr)
            dx = -fxr / dfdxr 
            xr = xr + dx
    
            i += 1
            epsA.append(abs(dx/xr))
            
    return xr, i, epsA

def fixed_point(x0, g):
    """
    fixed_point root finding
    Function takes in three input arguments: 
            x0 = initial guess
            g = function trying to find the root
     and outputs three arguments:
            xr = root 
            epsA = error
            iter = number of interations 
    """
            
    epsS = 1* 10**(-5) #error 
    epsA = 1    # intitialize error
    i = 0
    maxiter = 1000000  
    
    xr = x0 #intitalize root to guess
    
    while epsA > epsS and i < maxiter:
      
      xrold = xr
      xr = g(xrold)
      i = i+1
      
      if xr != 0:
          epsA = abs((xr-xrold)/xr)
      
    return xr, epsA, i






