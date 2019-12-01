#====================================================================================
#-----------------------     Jason Rampe 2 Attractor     ----------------------------
#====================================================================================

#-------------------     X = cos(Y * b) + c * cos(X * b)     ------------------------
#-------------------     Y = cos(X * a) + d * cos(Y * a)     ------------------------

#====================================================================================

import numpy as np
import pandas as pd
import panel as pn
import datashader as ds
from numba import jit
from datashader import transfer_functions as tf
from colorcet import palette_n

#------------------------------------------------------------------------------------

ps = {k:p[::-1] for k, p in palette_n.items()}

pn.extension()

#------------------------------------------------------------------------------------

@jit(nopython=True)
def JasonRampe2_trajectory(a, b, c, d, x0, y0, n):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    
    for i in np.arange(n-1):
        x[i+1] = np.cos(y[i] * b) + c * np.cos(x[i] * b)
        y[i+1] = np.cos(x[i] * a) + d * np.cos(y[i] * a)
        
    return x, y

#------------------------------------------------------------------------------------

def JasonRampe2_plot(a=2.29, b=-3.14, c=2.42, d=0.18, n=1000000, colormap=ps['kgy']):
    
    cvs = ds.Canvas(plot_width=800, plot_height=800)
    x, y = JasonRampe2_trajectory(a, b, c, d, 1, 1, n)
    agg = cvs.points(pd.DataFrame({'x':x, 'y':y}), 'x', 'y')
    
    return tf.shade(agg, cmap=colormap)

#------------------------------------------------------------------------------------

pn.interact(JasonRampe2_plot, n=(1,10000000), colormap=ps)

#------------------------------------------------------------------------------------

# The value of this attractor can be changed freely.
# Try it in the jupyter notebook.
