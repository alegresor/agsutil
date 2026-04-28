__version__ = "0.1"

from .algos import (
    lm_opt,
    minres,
    minres_qlp_cs,
    )
from .utils import (
    print_data_signatures,
    Timer, 
    to_unitary_qr,
    to_unitary_expskewh,
    from_unitary_expskewh,
    get_torch_rng,
    logmultinomialcoeff,
    multinomialcoeff,
    logfactorial,
    factorial,
    logcomb,
    comb,
    enumerate_sums,
    )
from .plots import (
    mpl_setup,
    set_aspect,
    )
