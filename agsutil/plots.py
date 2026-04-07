import numpy as np 

def mpl_setup():
    r""" 
    Setup matplotlib default parameters
    
    Returns:
        mplparams (dict): matplotlib helpful parameters"""
    from matplotlib import pyplot 
    pyplot.style.use("seaborn-v0_8-whitegrid")
    PW = 30 # page width in inches
    FW = 30 # font size
    COLORS = ["xkcd:purple","xkcd:blue","xkcd:green","xkcd:red","xkcd:orange","xkcd:cyan","xkcd:brown","xkcd:yellow"]
    LINESTYLES = ['solid','dotted','dashed','dashdot',(0, (1, 1))]
    pyplot.rcParams['xtick.labelsize'] = FS
    pyplot.rcParams['ytick.labelsize'] = FS
    pyplot.rcParams['ytick.labelsize'] = FS
    pyplot.rcParams['axes.titlesize'] = FS
    pyplot.rcParams['figure.titlesize'] = FS
    pyplot.rcParams["axes.labelsize"] = FS
    pyplot.rcParams['legend.fontsize'] = FS
    pyplot.rcParams['font.size'] = FS
    pyplot.rcParams['lines.linewidth'] = 5
    pyplot.rcParams['lines.markersize'] = 15
    mplparams = {
        "PW": PW,
        "FS": FS,
        "COLORS": COLORS,
        "LINESTYLES": LINESTYLES,
    }
    return mplparams

def set_aspects(ax, ratio=1):
    r"""
    Set aspect ratio of the ax. 

    Args:
        ax (Axes): axes to set the aspect ratio of. 
        ratio (float): positive aspect ratio for the axis
    """
    assert ratio>0
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    aspect = ratio*(xmax-xmin)/(ymax-ymin)
    ax.set_aspect(aspect)
