import numpy as np
import matplotlib.pyplot as plt

def Uncertainty_Plot(Title, x, y, uncert=[], x_label='', y_label='', scheme=1, filename=""):
    
      
    fig, ax = plt.subplots()

    Add_Uncertainty_Plot(ax, x, y, uncert, scheme)
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(Title)

    fig.tight_layout()
    
    if filename != "":
        plt.savefig(fname=filename)
        
    plt.show()
            
def Add_Uncertainty_Plot(axis, x, y, uncert=[], scheme=1):
    
    if scheme == 1:
        color1 = 'darkblue'
        color2 = 'skyblue'
    
    elif scheme == 2:
        color1 = 'red'
        color2 = 'orange'
        
    elif scheme == 3:
        color1 = 'green'
        color2 = 'palegreen'
        
    elif scheme == 4:
        color1 = 'black'
        color2 = 'silver'
        

    if x.size == y.size:
        
        y_values = y
        y_upper  = y + uncert
        y_lower  = y - uncert
        
    else:
        
        y_values = y[:,0]
        y_upper  = y[:,0] + y[:,1]
        y_lower  = y[:,0] - y[:,1]
        

    
    axis.plot(x, y_values, color=color1, linewidth=3)
    axis.plot(x, y_upper,  color=color2, linewidth=1)
    axis.plot(x, y_lower,  color=color2, linewidth=1)
        
           
            

        