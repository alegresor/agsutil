import torch 
import time

def print_data_signatures(data, name="data", print_devices=False, print_dtypes=False, verbose_indent=4):
    r""" 
    Print data shapes and (optionally) devices. 

    Args: 
        data (dict): Dictiony with items that are tensors or dictionaries of tensors. 
        print_devices (bool): If `True`, also print the device. 
        print_dtypes (bool): If `True`, also print the dtypes. 
        verbose_indent (int): Non-negative number of indentation spaces for logging.

    Examples:
        >>> data = {
        ...     "a": torch.rand(2,3,4),
        ...     "b": torch.rand(3,4,5),
        ...     "subdata": {
        ...         "aa": torch.rand(2,3),
        ...         "bb": torch.rand(2,3),
        ...         "subnontensor": ["ags",7,7,7],
        ...         },
        ...     "nontensor": [7,7,7,"ags"]
        ...     }
        >>> print_data_signatures(data,print_devices=True,print_dtypes=True,verbose_indent=0)
        data['a'].shape = (2, 3, 4) on device = cpu with dtype = torch.float64
        data['b'].shape = (3, 4, 5) on device = cpu with dtype = torch.float64
        data['subdata']
            data['subdata']['aa'].shape = (2, 3) on device = cpu with dtype = torch.float64
            data['subdata']['bb'].shape = (2, 3) on device = cpu with dtype = torch.float64
            data['subdata']['subnontensor'] a list of length 4
        data['nontensor'] a list of length 4
    """ 
    for key,val in data.items():
        if isinstance(val,torch.Tensor):
            _s = "%s['%s'].shape = %s"%(name,key,str(tuple(data[key].shape)))
            if print_devices:
                _s += " on device = %s"%str(data[key].device)
            if print_dtypes:
                _s += " with dtype = %s"%str(data[key].dtype)
            print(" "*verbose_indent+_s)
        elif isinstance(val,dict):
            print(" "*verbose_indent+"data['%s']"%key)
            for kkey,vval in val.items():
                if isinstance(vval,torch.Tensor):
                    _s = "%s['%s']['%s'].shape = %s"%(name,key,kkey,str(tuple(data[key][kkey].shape)))
                    if print_devices:
                        _s += " on device = %s"%str(data[key][kkey].device)
                    if print_dtypes:
                        _s += " with dtype = %s"%str(data[key][kkey].dtype)
                    print(" "*(verbose_indent+4)+_s)
                elif isinstance(vval,list):
                    print(" "*(verbose_indent+4)+"%s['%s']['%s'] a list of length %d"%(name,key,kkey,len(data[key][kkey])))
                else:
                    print(" "*(verbose_indent+4)+"%s['%s']['%s'] = %s"%(name,key,kkey,str(data[key][kkey])))
        elif isinstance(val,list):
            print(" "*verbose_indent+"%s['%s'] a list of length %d"%(name,key,len(data[key])))
        else:
            print(" "*verbose_indent+"%s['%s'] = %s"%(name,key,str(data[key])))

class Timer():
    r"""
    Timer compatible with CPU and GPU operations. 
    """

    def __init__(self, device):
        r"""
        Args:
            device (Union[torch.device,str]): device to perform timing for. 

                - For CPU devices, `time.perf_counter()` is used. 
                - For CUDA and MPS GPU devices, `torch.{cuda,mps}.Event(enable_timing=True)` is used. 
        """

        device_str = str(device) 
        if "cpu" in device_str:
            self.torch_backend = torch.cpu 
        elif "cuda" in device_str:
            self.torch_backend = torch.cuda 
        elif "mps" in device_str:
            self.torch_backend = torch.mps 
        else:
            raise Exception("undetected device = %s, should have 'cpu', 'cuda', or 'mps' in it."%device_str)
    def tic(self):
        r"""
        Start the stopwatch.
        """
        if self.torch_backend==torch.cpu:
            self.t0 = time.perf_counter()
        else:
            self.torch_backend.empty_cache()
            self.t0 = self.torch_backend.Event(enable_timing=True)
            self.tend = self.torch_backend.Event(enable_timing=True)
            self.t0.record()
    def toc(self):
        r"""
        Lap the stopwatch. 

        Returns: 
            tdelta (float): time elapsed between the start of the stopwatch and the current lap.
        """
        if self.torch_backend==torch.cpu:
            tdelta = time.perf_counter()-self.t0
        else:
            self.tend.record()
            self.torch_backend.synchronize()
            tdelta = self.t0.elapsed_time(self.tend)/1000
        return float(tdelta)

def logcomb(n,k):
    r"""
    $\log \binom{n}{k}$
    
    Args:
        n (torch.Tensor): $n$.
        k (torch.Tensor): $k$.

    Returns:
        nchoosek (torch.Tensor): $\log \binom{n}{k}$.

    Examples:
        >>> logcomb(torch.arange(1,8)[:,None],torch.arange(1,6)[None,:])
        tensor([[0.0000,   -inf,   -inf,   -inf,   -inf],
                [0.6931, 0.0000,   -inf,   -inf,   -inf],
                [1.0986, 1.0986, 0.0000,   -inf,   -inf],
                [1.3863, 1.7918, 1.3863, 0.0000,   -inf],
                [1.6094, 2.3026, 2.3026, 1.6094, 0.0000],
                [1.7918, 2.7081, 2.9957, 2.7081, 1.7918],
                [1.9459, 3.0445, 3.5553, 3.5553, 3.0445]])
        >>> torch.exp(logcomb(torch.arange(1,8)[:,None],torch.arange(1,6)[None,:])).to(int)
        tensor([[ 1,  0,  0,  0,  0],
                [ 2,  1,  0,  0,  0],
                [ 2,  2,  1,  0,  0],
                [ 4,  6,  4,  1,  0],
                [ 4,  9,  9,  4,  1],
                [ 6, 15, 20, 15,  6],
                [ 7, 21, 35, 35, 21]])
    """ 
    assert (n>=0).all()
    assert (k>=0).all()
    # assert (k<=n).all()
    return torch.lgamma(n+1)-torch.lgamma(k+1)-torch.lgamma(n-k+1)

def to_unitary_expskewh(theta, n, complex_case=False):
    r"""
    Transform to a unitary matrix.
    
    Args:
        theta (torch.Tensor): With `theta.size(-1) == n*(n-1)//2` in the real case and `theta.size(-1) == n**2` in the complex case.
        complex_case (bool): If `True`, parameterize a complex matrix, otherwise a real one. 
    
    Returns:
        Q (torch.Tensor): With shape `(*theta.shape[:-1],n,n)` orthonormal matrices
    
    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

    Single matrix
        
        >>> n = 5
        >>> theta = torch.rand(n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n)
        >>> Q 
        tensor([[ 0.5027, -0.1038, -0.2508,  0.3769,  0.7291],
                [-0.6870,  0.3679,  0.2777,  0.3454,  0.4430],
                [-0.2330, -0.7975,  0.3425, -0.2999,  0.3201],
                [-0.4638, -0.3141, -0.8037,  0.1779, -0.0934],
                [-0.0769,  0.3453, -0.3111, -0.7855,  0.4013]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q),torch.eye(n))
        True

    Single complex matrix
        
        >>> n = 4
        >>> theta = torch.rand(n**2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n,complex_case=True)
        >>> Q 
        tensor([[ 0.1358-0.3038j, -0.4798+0.2431j,  0.1105+0.2308j, -0.2229+0.6963j],
                [-0.4395+0.6202j,  0.1556-0.1974j, -0.1838-0.0838j, -0.1181+0.5516j],
                [-0.4750+0.0762j, -0.7622+0.0020j, -0.2319-0.0263j,  0.2346-0.2795j],
                [-0.0745-0.2729j,  0.2531+0.0496j, -0.7624+0.5079j,  0.1119+0.0410j]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q.conj()),torch.eye(n,dtype=torch.complex128))
        True

    Two matrices
        
        >>> n = 3
        >>> theta = torch.rand(2,n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n)
        >>> Q 
        tensor([[[ 0.3004,  0.6612,  0.6874],
                 [-0.7630,  0.5990, -0.2428],
                 [-0.5723, -0.4516,  0.6845]],
        <BLANKLINE>
                [[ 0.4583,  0.7552,  0.4687],
                 [-0.7977,  0.5820, -0.1579],
                 [-0.3921, -0.3015,  0.8691]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q),torch.eye(n))
        True
    
    Two complex matrices
        
        >>> n = 3
        >>> theta = torch.rand(2,n**2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n,complex_case=True)
        >>> Q 
        tensor([[[ 0.5591+0.3745j, -0.4342+0.5076j, -0.1283+0.2907j],
                 [-0.4802+0.4013j,  0.5099+0.4179j, -0.2657+0.3211j],
                 [-0.3843+0.0887j, -0.2972+0.1756j,  0.8185+0.2353j]],
        <BLANKLINE>
                [[ 0.4860+0.5953j, -0.1642+0.0458j, -0.3162+0.5295j],
                 [-0.5638+0.0805j,  0.3753+0.0607j,  0.2470+0.6856j],
                 [-0.2898+0.0325j, -0.8621+0.2884j,  0.2790+0.1037j]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q.conj()),torch.eye(n,dtype=torch.complex128))
        True

    Batch support

        >>> n = 10
        >>> theta = torch.rand(2,3,4,n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n)
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q),torch.eye(n))
        True
        >>> theta = torch.rand(2,3,4,n**2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n,complex_case=True)
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q.conj()),torch.eye(n,dtype=torch.complex128))
        True
    """
    batch_shape = tuple(theta.shape[:-1])
    batch_ones = torch.ones(batch_shape,dtype=int,device=theta.device)
    iut = torch.triu_indices(n,n,offset=1,device=theta.device)
    iutf = torch.einsum("...,i->...i",batch_ones,iut[0]*n+iut[1])
    iltf = torch.einsum("...,i->...i",batch_ones,iut[1]*n+iut[0])
    if complex_case:
        assert theta.size(-1)==(n**2)
        complex_dtype = (theta+0j).dtype
        H = torch.zeros((*batch_shape,n*n),dtype=complex_dtype,device=theta.device)
        idiag = torch.arange(n,device=theta.device)
        idiagf = torch.einsum("...,i->...i",batch_ones,idiag*n+idiag)
        diag = 1j*theta[...,:n].to(complex_dtype)
        off_real,off_imag = theta[...,n:n+(n**2-n)//2],theta[...,n+(n**2-n)//2:]
        off_vals = off_real.to(complex_dtype)+1j*off_imag.to(complex_dtype)
        H = H.scatter_add(-1,idiagf,diag)
        H = H.scatter_add(-1,iutf,off_vals)
        H = H.scatter_add(-1,iltf,-off_vals.conj())
        H = H.reshape((*batch_shape,n,n))
    else:
        assert theta.size(-1)==(n*(n-1)//2)
        H = torch.zeros((*batch_shape,n*n),dtype=theta.dtype,device=theta.device)
        H = H.scatter_add(-1,iutf,theta)
        H = H.scatter_add(-1,iltf,-theta)
        H = H.reshape((*batch_shape,n,n))
    return torch.matrix_exp(H)

def logm_unitary(U):
    vals, vecs = torch.linalg.eig(U.to((U+0j).dtype))
    log_vals = torch.log(vals)
    return torch.einsum("...ij,...j,...kj->...ik", vecs,log_vals,vecs.conj())

def from_unitary_expskewh(Q, complex_case=False):
    r"""
    Transform from a unitary matrix.
    
    Args:
        Q (torch.Tensor): With shape `(*theta.shape[:-1],n,n)` orthonormal matrices
        complex_case (bool): If `True`, parameterize a complex matrix, otherwise a real one. 
    
    Returns:
        theta (torch.Tensor): With `theta.size(-1) == n*(n-1)//2` in the real case and `theta.size(-1) == n**2` in the complex case.  
            Note that this `theta` is not unique, as shown in the following doctests.
    
    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

    Single matrix
        
        >>> n = 10
        >>> theta = torch.rand(n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n)
        >>> theta2 = from_unitary_expskewh(Q)
        >>> torch.allclose(theta,theta2)
        False
        >>> Q2 = to_unitary_expskewh(theta2,n)
        >>> torch.allclose(Q,Q2)
        True

    Single complex matrix
        
        >>> n = 10
        >>> theta = torch.rand(n**2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n,complex_case=True)
        >>> theta2 = from_unitary_expskewh(Q,complex_case=True)
        >>> torch.allclose(theta,theta2)
        False
        >>> Q2 = to_unitary_expskewh(theta2,n,complex_case=True)
        >>> torch.allclose(Q,Q2)
        True

    Two matrices
        
        >>> n = 10
        >>> theta = torch.rand(2,n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n)
        >>> theta2 = from_unitary_expskewh(Q)
        >>> torch.allclose(theta,theta2)
        False
        >>> Q2 = to_unitary_expskewh(theta2,n)
        >>> torch.allclose(Q,Q2)
        True

    Two complex matrices
        
        >>> n = 3
        >>> theta = torch.rand(2,n**2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n,complex_case=True)
        >>> theta2 = from_unitary_expskewh(Q,complex_case=True)
        >>> Q2 = to_unitary_expskewh(theta2,n,complex_case=True)
        >>> torch.allclose(Q,Q2)
        True
    
    Batch support

        >>> n = 10
        >>> theta = torch.rand(2,3,4,n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n)
        >>> theta2 = from_unitary_expskewh(Q)
        >>> Q2 = to_unitary_expskewh(theta2,n)
        >>> torch.allclose(Q,Q2)
        True
        >>> theta = torch.rand(2,3,4,n**2,generator=rng)
        >>> Q = to_unitary_expskewh(theta,n,complex_case=True)
        >>> theta2 = from_unitary_expskewh(Q,complex_case=True)
        >>> Q2 = to_unitary_expskewh(theta2,n,complex_case=True)
        >>> torch.allclose(Q,Q2)
        True
    """
    n = Q.size(-1)
    L = logm_unitary(Q)
    iut = torch.triu_indices(n,n,offset=1,device=Q.device)
    if complex_case:
        idiag = torch.arange(n,device=Q.device)
        diag = L[...,idiag,idiag].imag
        off_diag = L[...,iut[0],iut[1]]
        theta = torch.cat([diag,off_diag.real,off_diag.imag],dim=-1)
    else:
        theta = L[...,iut[0],iut[1]].real
    return theta

def to_unitary_householder(theta, n, complex_case=False):
    r"""
    Transform to a unitary matrix.
    
    Args:
        theta (torch.Tensor): With `theta.size(-1) == n*(n-1)//2` in the real case and `theta.size(-1) == n**2` in the complex case.
        complex_case (bool): If `True`, parameterize a complex matrix, otherwise a real one. 
    
    Returns:
        Q (torch.Tensor): With shape `(*theta.shape[:-1],n,n)` orthonormal matrices
    
    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

    Single matrix
        
        >>> n = 5
        >>> theta = torch.rand(n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_householder(theta,n)
        >>> Q 
        tensor([[ 0.5027, -0.1038, -0.2508,  0.3769,  0.7291],
                [-0.6870,  0.3679,  0.2777,  0.3454,  0.4430],
                [-0.2330, -0.7975,  0.3425, -0.2999,  0.3201],
                [-0.4638, -0.3141, -0.8037,  0.1779, -0.0934],
                [-0.0769,  0.3453, -0.3111, -0.7855,  0.4013]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q),torch.eye(n))
        True

    Single complex matrix
        
        >>> n = 4
        >>> theta = torch.rand(n**2,generator=rng)
        >>> Q = to_unitary_householder(theta,n,complex_case=True)
        >>> Q 
        tensor([[ 0.1358-0.3038j, -0.4798+0.2431j,  0.1105+0.2308j, -0.2229+0.6963j],
                [-0.4395+0.6202j,  0.1556-0.1974j, -0.1838-0.0838j, -0.1181+0.5516j],
                [-0.4750+0.0762j, -0.7622+0.0020j, -0.2319-0.0263j,  0.2346-0.2795j],
                [-0.0745-0.2729j,  0.2531+0.0496j, -0.7624+0.5079j,  0.1119+0.0410j]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q.conj()),torch.eye(n,dtype=torch.complex128))
        True

    Two matrices
        
        >>> n = 3
        >>> theta = torch.rand(2,n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_householder(theta,n)
        >>> Q 
        tensor([[[ 0.3004,  0.6612,  0.6874],
                 [-0.7630,  0.5990, -0.2428],
                 [-0.5723, -0.4516,  0.6845]],
        <BLANKLINE>
                [[ 0.4583,  0.7552,  0.4687],
                 [-0.7977,  0.5820, -0.1579],
                 [-0.3921, -0.3015,  0.8691]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q),torch.eye(n))
        True
    
    Two complex matrices
        
        >>> n = 3
        >>> theta = torch.rand(2,n**2,generator=rng)
        >>> Q = to_unitary_householder(theta,n,complex_case=True)
        >>> Q 
        tensor([[[ 0.5591+0.3745j, -0.4342+0.5076j, -0.1283+0.2907j],
                 [-0.4802+0.4013j,  0.5099+0.4179j, -0.2657+0.3211j],
                 [-0.3843+0.0887j, -0.2972+0.1756j,  0.8185+0.2353j]],
        <BLANKLINE>
                [[ 0.4860+0.5953j, -0.1642+0.0458j, -0.3162+0.5295j],
                 [-0.5638+0.0805j,  0.3753+0.0607j,  0.2470+0.6856j],
                 [-0.2898+0.0325j, -0.8621+0.2884j,  0.2790+0.1037j]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q.conj()),torch.eye(n,dtype=torch.complex128))
        True

    Batch support

        >>> n = 10
        >>> theta = torch.rand(2,3,4,n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_householder(theta,n)
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q),torch.eye(n))
        True
        >>> theta = torch.rand(2,3,4,n**2,generator=rng)
        >>> Q = to_unitary_householder(theta,n,complex_case=True)
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",Q,Q.conj()),torch.eye(n,dtype=torch.complex128))
        True
    """
    batch_shape = tuple(theta.shape[:-1])
    batch_ones = torch.ones(batch_shape,dtype=int,device=theta.device)
    iut = torch.triu_indices(n,n,offset=1,device=theta.device)
    iutf = torch.einsum("...,i->...i",batch_ones,iut[0]*n+iut[1])
    iltf = torch.einsum("...,i->...i",batch_ones,iut[1]*n+iut[0])
    if complex_case:
        assert theta.size(-1)==(n**2)
        complex_dtype = (theta+0j).dtype
        H = torch.zeros((*batch_shape,n*n),dtype=complex_dtype,device=theta.device)
        idiag = torch.arange(n,device=theta.device)
        idiagf = torch.einsum("...,i->...i",batch_ones,idiag*n+idiag)
        diag = 1j*theta[...,:n].to(complex_dtype)
        off_real,off_imag = theta[...,n:n+(n**2-n)//2],theta[...,n+(n**2-n)//2:]
        off_vals = off_real.to(complex_dtype)+1j*off_imag.to(complex_dtype)
        H = H.scatter_add(-1,idiagf,diag)
        H = H.scatter_add(-1,iutf,off_vals)
        H = H.scatter_add(-1,iltf,-off_vals.conj())
        H = H.reshape((*batch_shape,n,n))
    else:
        assert theta.size(-1)==(n*(n-1)//2)
        H = torch.zeros((*batch_shape,n*n),dtype=theta.dtype,device=theta.device)
        H = H.scatter_add(-1,iutf,theta)
        H = H.scatter_add(-1,iltf,-theta)
        H = H.reshape((*batch_shape,n,n))
    return torch.matrix_exp(H)

def from_unitary_householder(Q, complex_case=False):
    r"""
    Transform from a unitary matrix.
    
    Args:
        Q (torch.Tensor): With shape `(*theta.shape[:-1],n,n)` orthonormal matrices
        complex_case (bool): If `True`, parameterize a complex matrix, otherwise a real one. 
    
    Returns:
        theta (torch.Tensor): With `theta.size(-1) == n*(n-1)//2` in the real case and `theta.size(-1) == n**2` in the complex case.  
            Note that this `theta` is not unique, as shown in the following doctests.
    
    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

    Single matrix
        
        >>> n = 10
        >>> theta = torch.rand(n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_householder(theta,n)
        >>> theta2 = from_unitary_householder(Q)
        >>> torch.allclose(theta,theta2)
        False
        >>> Q2 = to_unitary_householder(theta2,n)
        >>> torch.allclose(Q,Q2)
        True

    Single complex matrix
        
        >>> n = 10
        >>> theta = torch.rand(n**2,generator=rng)
        >>> Q = to_unitary_householder(theta,n,complex_case=True)
        >>> theta2 = from_unitary_householder(Q,complex_case=True)
        >>> torch.allclose(theta,theta2)
        False
        >>> Q2 = to_unitary_householder(theta2,n,complex_case=True)
        >>> torch.allclose(Q,Q2)
        True

    Two matrices
        
        >>> n = 10
        >>> theta = torch.rand(2,n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_householder(theta,n)
        >>> theta2 = from_unitary_householder(Q)
        >>> torch.allclose(theta,theta2)
        False
        >>> Q2 = to_unitary_householder(theta2,n)
        >>> torch.allclose(Q,Q2)
        True

    Two complex matrices
        
        >>> n = 3
        >>> theta = torch.rand(2,n**2,generator=rng)
        >>> Q = to_unitary_householder(theta,n,complex_case=True)
        >>> theta2 = from_unitary_householder(Q,complex_case=True)
        >>> Q2 = to_unitary_householder(theta2,n,complex_case=True)
        >>> torch.allclose(Q,Q2)
        True
    
    Batch support

        >>> n = 10
        >>> theta = torch.rand(2,3,4,n*(n-1)//2,generator=rng)
        >>> Q = to_unitary_householder(theta,n)
        >>> theta2 = from_unitary_householder(Q)
        >>> Q2 = to_unitary_householder(theta2,n)
        >>> torch.allclose(Q,Q2)
        True
        >>> theta = torch.rand(2,3,4,n**2,generator=rng)
        >>> Q = to_unitary_householder(theta,n,complex_case=True)
        >>> theta2 = from_unitary_householder(Q,complex_case=True)
        >>> Q2 = to_unitary_householder(theta2,n,complex_case=True)
        >>> torch.allclose(Q,Q2)
        True
    """
    n = Q.size(-1)
    L = logm_unitary(Q)
    iut = torch.triu_indices(n,n,offset=1,device=Q.device)
    if complex_case:
        idiag = torch.arange(n,device=Q.device)
        diag = L[...,idiag,idiag].imag
        off_diag = L[...,iut[0],iut[1]]
        theta = torch.cat([diag,off_diag.real,off_diag.imag],dim=-1)
    else:
        theta = L[...,iut[0],iut[1]].real
    return theta

