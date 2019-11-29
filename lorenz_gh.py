#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is the Lorenz model solved with RK4.

Author: Mateo
Date: 15-Mar-2017
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def lorenz(s, b, r, xi, yi, zi):
    """Lorenz model"""
    # integration time
    ti = 0.0
    tf = 50.0
    dt = 0.02
    nsteps = int((tf-ti)/dt)
    
    vt = np.zeros(nsteps)
    vx = np.zeros(nsteps)
    vy = np.zeros(nsteps)
    vz = np.zeros(nsteps)

    t = ti
    x = xi
    y = yi
    z = zi

    for i in range(nsteps):
        vt[i] = t
        vx[i] = x
        vy[i] = y
        vz[i] = z
        
        k1x = -s*(x-y)
        k2x = -s*((x+0.5*k1x*dt)-y)
        k3x = -s*((x+0.5*k2x*dt)-y)
        k4x = -s*((x+k3x*dt)-y)
        dxdt = (k1x+2*k2x+2*k3x+k4x)/6
        x = x+dxdt*dt

        k1y = x*(r-z)-y
        k2y = x*(r-z)-(y+0.5*k1y*dt)
        k3y = x*(r-z)-(y+0.5*k2y*dt)
        k4y = x*(r-z)-(y+k3y*dt)
        dydt = (k1y+2*k2y+2*k3y+k4y)/6
        y = y+dydt*dt

        k1z = x*y-b*z
        k2z = x*y-b*(z+0.5*k1z*dt)
        k3z = x*y-b*(z+0.5*k2z*dt)
        k4z = x*y-b*(z+k3z*dt)
        dzdt = (k1z+2*k2z+2*k3z+k4z)/6
        z = z+dzdt*dt

        t = ti+i*dt
    return vt,vx,vy,vz,nsteps

# parameters
s = 10.0
b = 2.67
r = 28.0

vt1,vx1,vy1,vz1,nsteps = lorenz(s, b, r, 12.0,12.0,12.0)
vt2,vx2,vy2,vz2,nsteps = lorenz(s, b, r, 12.001,12.0,12.0)
    
cmap = mpl.cm.BrBG

# 3d phase diagram
for i in range(nsteps):
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(min(vx1),max(vx1)+5)
    ax.set_ylim(min(vy1),max(vy1)+5)
    ax.set_zlim(min(vz1),max(vz1)+5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz model with r = %3.1f" % r)
    ax.view_init(30,i)
    ax.plot(vx1[:i+1],vy1[:i+1],vz1[:i+1],color='Goldenrod')
    ax.plot(vx2[:i+1],vy2[:i+1],vz2[:i+1],color='DodgerBlue')
    ax.plot([vx1[i]],[vy1[i]],[vz1[i]],'o',color='Goldenrod')
    ax.plot([vx2[i]],[vy2[i]],[vz2[i]],'o',color='DodgerBlue')
    plt.savefig("images/chaos/lorenz_%05d.png" % i)
    plt.clf()
