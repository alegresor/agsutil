__version__ = "0.2"

from .algos import (
    lm_opt,
    pcg,
    minres,
    minres_qlp_cs,
    )
from .autograd import (
    gradb,
    jacfwdb,
    jacrevb,
    jvpb,
    vjpb
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
    len_enumerate_sums,
    enumerate_partitions,
    icdf_std_normal,
    )
from .plots import (
    mpl_setup,
    set_aspect,
    )
