import numpy as np 

def mpl_setup():
    r""" 
    Setup matplotlib default parameters
    
    Returns:
        mplparams (dict): matplotlib helpful parameters"""
    from matplotlib import pyplot 
    import seaborn as sns
    pyplot.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind") # Options: "deep", "muted", "pastel", "bright", "dark", "colorblind
    COLORS = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
    PW = 30 # page width in inches
    FS = 30 # font size
    LINESTYLES = [
        'solid',
        'dotted',
        'dashdot',
        'dashed',
        (0, (3, 5, 1, 5, 1, 5)),
        (5, (10, 3)),
        (0, (1, 1)),
        (0, (1, 10)),
        (0, (1, 5)),
        (0, (5, 5)),
        ]
    MARKERS = [
        "o",
        "s",
        "D",
        "P",
        "^",
        "v",
        "<",
        ">",
        "$a$",
        "$b$",
        "$c$",
        "$d$",
        ]
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
        "MARKERS": MARKERS,
        }
    return mplparams

def set_aspect(ax, ratio=1):
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
