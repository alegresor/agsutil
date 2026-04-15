__version__ = "0.1"

from .algos import (
    lm_opt,
    minres,
    minres_qlp_cs,
    )
from .utils import (
    print_data_signatures,
    Timer, 
    )
from .plots import (
    mpl_setup,
    set_aspect,
    )
