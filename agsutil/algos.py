import torch 
import numpy as np
import warnings 
import time

class Timer():
    def __init__(self, device):
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
        if self.torch_backend==torch.cpu:
            self.t0 = time.perf_counter()
        else:
            self.torch_backend.empty_cache()
            self.t0 = self.torch_backend.Event(enable_timing=True)
            self.tend = self.torch_backend.Event(enable_timing=True)
            self.t0.record()
    def toc(self):
        if self.torch_backend==torch.cpu:
            tdelta = time.perf_counter()-self.t0
        else:
            self.tend.record()
            self.torch_backend.synchronize()
            tdelta = self.t0.elapsed_time(self.tend)/1000
        return tdelta

def print_data_signatures(data, show_device=False):
    for key,val in data.items():
        if isinstance(val,torch.Tensor):
            _s = "    data['%s'].shape = %s"%(key,str(tuple(data[key].shape)))
            if show_device:
                _s += " on device = %s"%str(data[key].device)
            print(_s)
        elif isinstance(val,dict):
            print("    data['%s']"%key)
            for kkey,vval in val.items():
                _s = "        data['%s']['%s'].shape = %s"%(key,kkey,str(tuple(data[key][kkey].shape)))
                if show_device:
                    _s += " on device = %s"%str(data[key][kkey].device)
                print(_s)
        else:
            print("    data['%s'] = %s"%(key,str(data[key])))

def lm_opt(
        f,
        theta0,
        ytrue,
        iters = 10,
        batch_dims = 0, 
        f_kwargs_vec = {},
        f_kwargs_no_vec = {},
        lam0 = 1e-6,
        alpha0 = 1e0,
        lam_factors = [[1/2,1,2],[1]],
        alpha_factors = [[1],[1/2,1,2]],
        vmap_chunk_size = None,
        jacfwd = True,
        verbose = False, 
        verbose_indent = 4,
        quantiles_losses = [0,1,5,10,25,40,50,60,75,90,95,99,100],
        quantiles_lams =   [0,1,5,10,25,40,50,60,75,90,95,99,100],
        quantiles_alphas = [0,1,5,10,25,40,50,60,75,90,95,99,100],
        verbose_quantiles_losses = [5,25,50,75,90],
        verbose_quantiles_lams =   [5,25,50,75,90],
        verbose_quantiles_alphas = [5,25,50,75,90],
        verbose_times = True, 
        warn = True,
        ):
    r"""
    Levenberg--Marquardt optimization 

    Args:
        f (func): Residual function. 
        theta0 (torch.Tensor): Initial guess for parameters $\theta$. 
        ytrue (torch.Tensor): True `y` values, i.e. `f(theta_true)`. 
        iters (int): Number of iterations. 
        batch_dims (int): Number of batch dimension. 
        f_kwargs_vec (dict): Keyword arguments to `f` which will be vectorized over the first dimension. 
        f_kwargs_no_vec (dict): Keyword arguments to `f` which will not be vectorized over the first dimension. 
        lam0 (float): Initial positive relaxation parameter $\lambda$.
        alpha0 (float): Initial positive step size $\alpha$.
        lam_factors (torch.Tensor): Either a 1D `torch.Tensor` or list of 1d `torch.Tensor`. 

            - Passing in a `float` for `lam_factors` is equivalent to passing in `torch.tensor([lam_factors])` on the correct device
            - If a 1D `torch.Tensor` for `lam_factors` will consider all `lam*lam_factors` options at each step. 
            - If a list of 1D `torch.Tensor`s are passed in for `lam_factors`, iterations will cycle through the list and then return to the start after exhausting the list.

        alpha_factors (torch.Tensor): Either a 1D `torch.Tensor` or list of 1d `torch.Tensor`. 

            - Passing in a `float` for `alpha_factors` is equivalent to passing in `torch.tensor([alpha_factors])` on the correct device
            - If a 1D `torch.Tensor` for `alpha_factors` will consider all `lam*alpha_factors` options at each step. 
            - If a list of 1D `torch.Tensor`s are passed in for `alpha_factors`, iterations will cycle through the list and then return to the start after exhausting the list. 
        
        vmap_chunk_size (int): Parameter `chunksize` to pass to `torch.vmap`.
        jacfwd (bool): If `True`, use `torch.func.jacfwd`, otherwise use `torch.func.jacrev`
        verbose (int): Controls logging verbosity
        
            - If True, perform logging. 
            - If a positive int, only log every verbose iterations. 
            - If None, set to a reasonable positive int based on the maximum number of iterations
            - If False, don't log. 
        
        verbose_indent (int): Positive number of indentation spaces for logging.
        quantiles_losses (list): Loss quantiles to record.
        quantiles_lams (list): $\lambda$ quantiles to record.
        quantiles_alphas (list): $\alpha$ quantiles to record.
        verbose_quantiles_losses (list): Loss quantiles to show in verbose log.
        verbose_quantiles_lams (list): $\lambda$ quantiles to show in verbose log.
        verbose_quantiles_alphas (list): $\alpha$ quantiles to show in verbose log.
        verbose_times (bool): If `False`, do not show the times in the verbose log. This is mostly for testing where timing is not reproducible. 
        warn (bool): If `False`, then suppress warnings.
    
    Returns:
        theta (torch.Tensor): Optimized parameters.
        data (dict): Iteration data.

    Examples:

        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

        >>> x = torch.rand((10,4,),generator=rng)
        >>> theta_true = torch.rand((4,),generator=rng)
        >>> ytrue = torch.exp((x*theta_true).sum(-1)) # (10,)
        >>> def f(theta):
        ...     yhat = torch.exp((x*theta[...,None,:]).sum(-1)) # (...,10)
        ...     return yhat
        >>> theta_hat,data = lm_opt(
        ...     f = f, 
        ...     theta0 = torch.rand_like(theta_true,generator=rng),
        ...     ytrue = ytrue,
        ...     iters = 3,
        ...     batch_dims = 0,
        ...     verbose = True,
        ...     verbose_times = False,
        ...     )
            iter i     | losses_quantiles                                          | lams_quantiles                                            | alphas_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        | 5         | 25        | 50        | 75        | 90        | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 2.3e+01   | 2.3e+01   | 2.3e+01   | 2.3e+01   | 2.3e+01   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 7.3e+00   | 7.3e+00   | 7.3e+00   | 7.3e+00   | 7.3e+00   | 2.0e-06   | 2.0e-06   | 2.0e-06   | 2.0e-06   | 2.0e-06   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            2          | 8.5e-02   | 8.5e-02   | 8.5e-02   | 8.5e-02   | 8.5e-02   | 2.0e-06   | 2.0e-06   | 2.0e-06   | 2.0e-06   | 2.0e-06   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            3          | 3.3e-05   | 3.3e-05   | 3.3e-05   | 3.3e-05   | 3.3e-05   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
        >>> torch.allclose(theta_hat,theta_true,atol=5e-2)
        True
        >>> print_data_signatures(data)
            data['theta'].shape = (4,)
            data['iterrange'].shape = (4,)
            data['times'].shape = (4,)
            data['thetas'].shape = (4, 4)
            data['losses'].shape = (4,)
            data['lams'].shape = (4,)
            data['alphas'].shape = (4,)
            data['losses_quantiles']
                data['losses_quantiles']['0'].shape = (4,)
                data['losses_quantiles']['1'].shape = (4,)
                data['losses_quantiles']['5'].shape = (4,)
                data['losses_quantiles']['10'].shape = (4,)
                data['losses_quantiles']['25'].shape = (4,)
                data['losses_quantiles']['40'].shape = (4,)
                data['losses_quantiles']['50'].shape = (4,)
                data['losses_quantiles']['60'].shape = (4,)
                data['losses_quantiles']['75'].shape = (4,)
                data['losses_quantiles']['90'].shape = (4,)
                data['losses_quantiles']['95'].shape = (4,)
                data['losses_quantiles']['99'].shape = (4,)
                data['losses_quantiles']['100'].shape = (4,)
            data['lams_quantiles']
                data['lams_quantiles']['0'].shape = (4,)
                data['lams_quantiles']['1'].shape = (4,)
                data['lams_quantiles']['5'].shape = (4,)
                data['lams_quantiles']['10'].shape = (4,)
                data['lams_quantiles']['25'].shape = (4,)
                data['lams_quantiles']['40'].shape = (4,)
                data['lams_quantiles']['50'].shape = (4,)
                data['lams_quantiles']['60'].shape = (4,)
                data['lams_quantiles']['75'].shape = (4,)
                data['lams_quantiles']['90'].shape = (4,)
                data['lams_quantiles']['95'].shape = (4,)
                data['lams_quantiles']['99'].shape = (4,)
                data['lams_quantiles']['100'].shape = (4,)
            data['alphas_quantiles']
                data['alphas_quantiles']['0'].shape = (4,)
                data['alphas_quantiles']['1'].shape = (4,)
                data['alphas_quantiles']['5'].shape = (4,)
                data['alphas_quantiles']['10'].shape = (4,)
                data['alphas_quantiles']['25'].shape = (4,)
                data['alphas_quantiles']['40'].shape = (4,)
                data['alphas_quantiles']['50'].shape = (4,)
                data['alphas_quantiles']['60'].shape = (4,)
                data['alphas_quantiles']['75'].shape = (4,)
                data['alphas_quantiles']['90'].shape = (4,)
                data['alphas_quantiles']['95'].shape = (4,)
                data['alphas_quantiles']['99'].shape = (4,)
                data['alphas_quantiles']['100'].shape = (4,)

        >>> x = torch.rand((3,3,3,2,2),generator=rng)
        >>> theta_true = torch.rand((4,4,2,2),generator=rng)
        >>> ytrue = torch.exp((x*theta_true[...,None,None,None,:,:]).sum((-2,-1))) # (4,4,3,3,3)
        >>> def f(theta):
        ...     yhat = torch.exp((x*theta[...,None,None,None,:,:]).sum((-2,-1))) # (...,3,3,3)
        ...     return yhat
        >>> theta_hat,data = lm_opt(
        ...     f = f, 
        ...     theta0 = torch.rand_like(theta_true,generator=rng),
        ...     ytrue = ytrue,
        ...     iters = 2,
        ...     batch_dims = 2,
        ...     lam_factors = [torch.tensor([1/4,1/2,1,2,4])],
        ...     alpha_factors = [torch.tensor([2/3,1,3/2])],
        ...     verbose = True,
        ...     verbose_times = False,
        ...     )
            iter i     | losses_quantiles                                          | lams_quantiles                                            | alphas_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        | 5         | 25        | 50        | 75        | 90        | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 5.5e+00   | 2.5e+01   | 5.3e+01   | 1.3e+02   | 5.1e+02   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 7.4e-02   | 6.4e-01   | 1.1e+00   | 2.2e+00   | 7.7e+01   | 2.5e-07   | 2.5e-07   | 2.5e-07   | 2.5e-07   | 4.0e-06   | 6.7e-01   | 6.7e-01   | 1.0e+00   | 1.1e+00   | 1.5e+00   
            2          | 1.0e-05   | 6.9e-04   | 1.8e-03   | 3.9e-03   | 1.2e+00   | 6.2e-08   | 6.2e-08   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 6.7e-01   | 6.7e-01   | 1.0e+00   | 1.1e+00   | 1.5e+00   
        >>> torch.allclose(theta_hat,theta_true,atol=5e-2)
        False
        >>> print_data_signatures(data)
            data['theta'].shape = (4, 4, 2, 2)
            data['iterrange'].shape = (3,)
            data['times'].shape = (3,)
            data['thetas'].shape = (3, 4, 4, 2, 2)
            data['losses'].shape = (3, 4, 4)
            data['lams'].shape = (3, 4, 4)
            data['alphas'].shape = (3, 4, 4)
            data['losses_quantiles']
                data['losses_quantiles']['0'].shape = (3,)
                data['losses_quantiles']['1'].shape = (3,)
                data['losses_quantiles']['5'].shape = (3,)
                data['losses_quantiles']['10'].shape = (3,)
                data['losses_quantiles']['25'].shape = (3,)
                data['losses_quantiles']['40'].shape = (3,)
                data['losses_quantiles']['50'].shape = (3,)
                data['losses_quantiles']['60'].shape = (3,)
                data['losses_quantiles']['75'].shape = (3,)
                data['losses_quantiles']['90'].shape = (3,)
                data['losses_quantiles']['95'].shape = (3,)
                data['losses_quantiles']['99'].shape = (3,)
                data['losses_quantiles']['100'].shape = (3,)
            data['lams_quantiles']
                data['lams_quantiles']['0'].shape = (3,)
                data['lams_quantiles']['1'].shape = (3,)
                data['lams_quantiles']['5'].shape = (3,)
                data['lams_quantiles']['10'].shape = (3,)
                data['lams_quantiles']['25'].shape = (3,)
                data['lams_quantiles']['40'].shape = (3,)
                data['lams_quantiles']['50'].shape = (3,)
                data['lams_quantiles']['60'].shape = (3,)
                data['lams_quantiles']['75'].shape = (3,)
                data['lams_quantiles']['90'].shape = (3,)
                data['lams_quantiles']['95'].shape = (3,)
                data['lams_quantiles']['99'].shape = (3,)
                data['lams_quantiles']['100'].shape = (3,)
            data['alphas_quantiles']
                data['alphas_quantiles']['0'].shape = (3,)
                data['alphas_quantiles']['1'].shape = (3,)
                data['alphas_quantiles']['5'].shape = (3,)
                data['alphas_quantiles']['10'].shape = (3,)
                data['alphas_quantiles']['25'].shape = (3,)
                data['alphas_quantiles']['40'].shape = (3,)
                data['alphas_quantiles']['50'].shape = (3,)
                data['alphas_quantiles']['60'].shape = (3,)
                data['alphas_quantiles']['75'].shape = (3,)
                data['alphas_quantiles']['90'].shape = (3,)
                data['alphas_quantiles']['95'].shape = (3,)
                data['alphas_quantiles']['99'].shape = (3,)
                data['alphas_quantiles']['100'].shape = (3,)
    """
    if warn and (not torch.get_default_dtype()==torch.float64): warnings.warn('''
            torch.get_default_dtype() = %s, but lm_opt often requires high precision updates. We recommend using:
                torch.set_default_dtype(torch.float64)'''%str(torch.get_default_dtype()))
    device = str(theta0.device)
    default_device = str(torch.get_default_device())
    assert iters%1==0, "iters should be an int"
    assert iters>=0
    assert callable(f) 
    assert batch_dims>=0
    assert isinstance(theta0,torch.Tensor)
    batch_shape = tuple(theta0.shape[:batch_dims])
    R = int(torch.tensor(batch_shape).prod())
    nonbatch_theta_dims = theta0.ndim-batch_dims
    nonbatch_theta_shape = tuple(theta0.shape[batch_dims:])
    nonbatch_y_dims = ytrue.ndim-batch_dims
    nonbatch_y_shape = tuple(ytrue.shape[batch_dims:])
    K = int(torch.tensor(nonbatch_y_shape).prod())
    T = int(torch.tensor(nonbatch_theta_shape).prod())
    if batch_dims==0:
        theta = theta0[None,...]
    else: # batch_dims>0:
        theta = theta0.flatten(end_dim=batch_dims-1)
    if batch_dims==0:
        ytrue = ytrue[None,...]
    else: # batch_dims>0:
        ytrue = ytrue.flatten(end_dim=batch_dims-1)
    assert isinstance(f_kwargs_vec,dict)
    assert isinstance(f_kwargs_no_vec,dict)
    f_kwargs_vec_names = list(f_kwargs_vec.keys())
    f_kwargs_vec_vals = []
    for key in f_kwargs_vec_names:
        assert f_kwargs_vec[key].shape[:batch_dims]==batch_shape, "f_kwargs_vec['%s'].shape[:%d] = %s but bs = %s"%(key,batch_dims,f_kwargs_vec[key].shape[:batch_dims])
        if batch_dims==0:
            f_kwargs_vec_vals.append(f_kwargs_vec[key][None,...])
        else: # batch_dims>0
            f_kwargs_vec_vals.append(f_kwargs_vec[key].flatten(end_dim=batch_dims-1))
    f_kwargs_vec_names = ["ytrue"]+f_kwargs_vec_names
    f_kwargs_vec_vals = [ytrue]+f_kwargs_vec_vals
    if verbose is None: 
        verbose = max(1,iters//20)
    assert lam0>0
    assert alpha0>0
    if np.isscalar(lam_factors):
        lam_factors = [torch.tensor([lam_factors],device=device)]
    elif isinstance(lam_factors,torch.Tensor):
        lam_factors = [lam_factors.to(device)]
    lam_factors = [torch.tensor(list(lam_factors[i])).to(device) for i in range(len(lam_factors))]
    assert isinstance(lam_factors,list)
    assert all(isinstance(lam_factors[i],torch.Tensor) for i in range(len(lam_factors)))
    assert all(lam_factors[i].ndim==1 for i in range(len(lam_factors)))
    if np.isscalar(alpha_factors):
        alpha_factors = [torch.tensor([alpha_factors],device=device)]
    elif isinstance(alpha_factors,torch.Tensor):
        alpha_factors = [alpha_factors.to(device)]
    alpha_factors = [torch.tensor(list(alpha_factors[i])).to(device) for i in range(len(alpha_factors))]
    assert isinstance(alpha_factors,list)
    assert all(isinstance(alpha_factors[i],torch.Tensor) for i in range(len(alpha_factors)))
    assert all(alpha_factors[i].ndim==1 for i in range(len(alpha_factors)))
    if len(alpha_factors)==1:
        alpha_factors = alpha_factors*len(lam_factors)
    if len(lam_factors)==1:
        lam_factors = lam_factors*len(alpha_factors)
    assert len(lam_factors)==len(alpha_factors)
    assert isinstance(quantiles_losses,list)
    assert all(0<=qt<=100 for qt in quantiles_losses)
    assert isinstance(quantiles_lams,list)
    assert all(0<=qt<=100 for qt in quantiles_lams)
    assert isinstance(quantiles_alphas,list)
    assert all(0<=qt<=100 for qt in quantiles_alphas)
    assert isinstance(verbose_quantiles_losses,list)
    assert all(qt in quantiles_losses for qt in verbose_quantiles_losses)
    assert isinstance(verbose_quantiles_lams,list)
    assert all(qt in quantiles_lams for qt in verbose_quantiles_lams)
    assert isinstance(verbose_quantiles_alphas,list)
    assert all(qt in quantiles_alphas for qt in verbose_quantiles_alphas)
    assert verbose%1==0
    assert verbose>=0 
    assert verbose_indent%1==0 
    assert verbose_indent>=0
    assert isinstance(verbose_times,bool)
    thetas = torch.nan*torch.ones((iters+1,*batch_shape,*nonbatch_theta_shape),device=default_device)
    times = torch.nan*torch.ones(iters+1,device=default_device)
    losses = torch.nan*torch.ones((iters+1,*batch_shape),device=default_device)
    lams = torch.nan*torch.ones((iters+1,*batch_shape),device=default_device)
    alphas = torch.nan*torch.ones((iters+1,*batch_shape),device=default_device)
    losses_quantiles = {str(qt):torch.nan*torch.ones(iters+1,device=default_device) for qt in quantiles_losses}
    lams_quantiles = {str(qt):torch.nan*torch.ones(iters+1,device=default_device) for qt in quantiles_lams}
    alphas_quantiles = {str(qt):torch.nan*torch.ones(iters+1,device=default_device) for qt in quantiles_alphas}
    def f_resid(theta, *f_kwargs_vec_vals):
        assert len(f_kwargs_vec_vals)==len(f_kwargs_vec_names)
        ytrue = f_kwargs_vec_vals[0]
        f_kwargs_vec = {f_kwargs_vec_names[i]:f_kwargs_vec_vals[i] for i in range(1,len(f_kwargs_vec_names))}
        yhat = f(theta,**f_kwargs_vec,**f_kwargs_no_vec)
        resid = yhat-ytrue
        return resid,(resid,yhat)
    assert isinstance(jacfwd,bool)
    if jacfwd:
        jac_ftilde = torch.func.jacfwd(f_resid,argnums=(0,),has_aux=True)
    else:
        jac_ftilde = torch.func.jacrev(f_resid,argnums=(0,),has_aux=True)
    vjac_ftilde = torch.func.vmap(jac_ftilde,in_dims=(0,)+(0,)*len(f_kwargs_vec_names),chunk_size=vmap_chunk_size)
    if warn and jacfwd and T>K: warnings.warn('''
        For T the number of inputs and K the number of outputs:
            torch.func.jacfwd performs best when T << K. 
            torch.func.jacrev performs best when K << T.
        You are using torch.func.jacfwd but T = %d > %d = K. 
        Try using torch.func.jacrev by setting jacfwd = False.'''%(T,K))
    if warn and (not jacfwd) and T<K: warnings.warn('''
        For T the number of inputs and K the number of outputs:
            torch.func.jacfwd performs best when T << K. 
            torch.func.jacrev performs best when K << T.
        You are using torch.func.jacrev but T = %d < %d = K. 
        Try using torch.func.jacrev by setting jacfwd = True.'''%(T,K))
    eyeT = torch.eye(T,device=device)
    Rrange = torch.arange(R,device=device)
    lam = lam0*torch.ones(R,device=device)
    alpha = alpha0*torch.ones(R,device=device)
    if verbose:
        _h_iter = "%-10s "%"iter i"
        _h_times = "| %-10s"%"times" if verbose_times else ""
        _s_losses_qt = ("| %-9s "*len(verbose_quantiles_losses))%tuple(str(qt) for qt in verbose_quantiles_losses)
        _s_lams_qt = ("| %-9s "*len(verbose_quantiles_lams))%tuple(str(qt) for qt in verbose_quantiles_lams)
        _s_alphas_qt = ("| %-9s "*len(verbose_quantiles_alphas))%tuple(str(qt) for qt in verbose_quantiles_alphas)
        _h_losses_qt = "| losses_quantiles"+" "*(len(_s_losses_qt)-len("| losses_quantiles"))
        _h_lams_qt   = "| lams_quantiles"  +" "*(len(_s_lams_qt)  -len("| lams_quantiles"))
        _h_alphas_qt = "| alphas_quantiles"+" "*(len(_s_alphas_qt)-len("| alphas_quantiles"))
        _h = _h_iter+_h_losses_qt+_h_lams_qt+_h_alphas_qt+_h_times
        _s = " "*len(_h_iter)+_s_losses_qt+_s_lams_qt+_s_alphas_qt+("|"+" "*(len(_h_times)-1) if verbose_times else " "*len(_h_times))
        print(" "*verbose_indent+_h)
        print(" "*verbose_indent+_s)
        print(" "*verbose_indent+"~"*len(_s))
    timer = Timer(device=device)
    timer.tic()
    for i in range(iters+1):
        if i==iters:
            _,(resid,yhat) = f_resid(theta,*f_kwargs_vec_vals)
        else:
            (Jfull,),(resid,yhat) = vjac_ftilde(theta,*f_kwargs_vec_vals)
        assert Jfull.shape==(R,*nonbatch_y_shape,*nonbatch_theta_shape)
        thetas[i] = theta.reshape(*batch_shape,*nonbatch_theta_shape).to(default_device)
        loss = (resid**2).flatten(start_dim=1).sum(-1)
        losses[i] = loss.reshape(batch_shape).to(default_device)
        lams[i] = lam.reshape(batch_shape).to(default_device)
        alphas[i] = alpha.reshape(batch_shape).to(default_device)
        for qt in quantiles_losses:
            losses_quantiles[str(qt)][i] = loss.nanquantile(qt/100).to(default_device)
        for qt in quantiles_lams:
            lams_quantiles[str(qt)][i] = lam.nanquantile(qt/100).to(default_device)
        for qt in quantiles_alphas:
            alphas_quantiles[str(qt)][i] = alpha.nanquantile(qt/100).to(default_device)
        times[i] = timer.toc()
        if verbose and (i%verbose==0 or i==iters):
            _s_iter = "%-10d "%i
            _s_losses_qt = ("| %-9.1e "*len(verbose_quantiles_losses))%tuple(losses_quantiles[str(qt)][i] for qt in verbose_quantiles_losses)
            _s_lams_qt = ("| %-9.1e "*len(verbose_quantiles_lams))%tuple(lams_quantiles[str(qt)][i] for qt in verbose_quantiles_lams)
            _s_alphas_qt = ("| %-9.1e "*len(verbose_quantiles_alphas))%tuple(alphas_quantiles[str(qt)][i] for qt in verbose_quantiles_alphas)
            _s_times = "| %-10.1f "%(times[i]) if verbose_times else ""
            print(" "*verbose_indent+_s_iter+_s_losses_qt+_s_lams_qt+_s_alphas_qt+_s_times)
        if i==iters: break
        J = Jfull.reshape((R,K,T))
        residf = resid.reshape((R,K))
        gamma = torch.einsum("rij,ri->rj",J,residf)
        JtJ = torch.einsum("rij,ril->rjl",J,J) # (R,T,T)
        lam_factors_i = lam_factors[i%len(lam_factors)]
        alpha_factors_i = alpha_factors[i%len(alpha_factors)]
        Q_lams = len(lam_factors_i)
        Q_alphas = len(alpha_factors_i)
        lams_try = lam_factors_i[:,None]*lam # (Q_lams,R)
        alphas_try = alpha_factors_i[:,None]*alpha # (Q_alphas,R)
        JtJplam = JtJ[None,:,:,:]+lams_try[:,:,None,None]*eyeT # (Q_lams,R,T,T)
        L,fails = torch.linalg.cholesky_ex(JtJplam,upper=False) # L.shape==(Q_lams,R,T,T) and fails.shape==(Q_lams,R)
        success = ~fails.to(bool) # (Q_lams,R)
        deltaf = torch.nan*torch.ones((Q_lams,R,T),device=device)
        gammas = torch.ones((Q_lams,1,1),device=device)*gamma[None,:,:] # (Q_lams,R,T)
        deltaf[success] = torch.linalg.solve_triangular(L[success].transpose(dim0=-2,dim1=-1),torch.linalg.solve_triangular(L[success],gammas[success,...,None],upper=False),upper=True)[...,0] # (Q_lam,R,T)
        thetasf = torch.ones((Q_lams,1,1),device=device)*theta.reshape((1,R,T)) # (Q_lam,R,T)
        thetasf_new = torch.nan*torch.ones((Q_alphas,Q_lams,R,T),device=device)
        thetasf_new[:,success] = thetasf[success]-alpha_factors_i[:,None,None]*deltaf[success]
        thetas_new = thetasf_new.reshape((Q_alphas,Q_lams,R,*nonbatch_theta_shape))
        f_kwargs_vec_vals_success = [(torch.ones((Q_alphas,Q_lams)+(1,)*f_kwargs_vec_vals[l].ndim,device=device)*f_kwargs_vec_vals[l][None,None,...])[:,success] for l in range(len(f_kwargs_vec_vals))]
        residf_new = torch.inf*torch.ones((Q_alphas,Q_lams,R,K),device=device)
        _,(resid_new_success,_) = f_resid(thetas_new[:,success],*f_kwargs_vec_vals_success)
        residf_new[:,success] = resid_new_success.reshape((Q_alphas,resid_new_success.size(1),K))
        losses_new = (residf_new**2).sum(-1)
        imin = losses_new.reshape((Q_alphas*Q_lams,R)).argmin(0) # (R,)
        imin_alpha,imin_lam = imin//Q_lams,imin%Q_lams
        lam_best_new = lams_try[imin_lam,Rrange] # (R,)
        alpha_best_new = alphas_try[imin_alpha,Rrange] # (R,)
        loss_best_new = losses_new[imin_alpha,imin_lam,Rrange] # (R,)
        thetas_best_new = thetas_new[imin_alpha,imin_lam,Rrange] # (R,*nonbatch_theta_shape)
        improved = loss_best_new<loss # (R,)
        lam[improved] = lam_best_new[improved]
        alpha[improved] = alpha_best_new[improved]
        theta[improved] = thetas_best_new[improved]
    theta = theta.reshape((*batch_shape,*nonbatch_theta_shape))
    data = {
        "theta": theta.to(default_device), 
        "iterrange": torch.arange(iters+1), 
        "times": times, 
        "thetas": thetas,
        "losses": losses,
        "lams": lams, 
        "alphas": alphas,
        "losses_quantiles": losses_quantiles,
        "lams_quantiles": lams_quantiles,
        "alphas_quantiles": alphas_quantiles,
        }
    return theta,data

def minres(
        A,
        B,
        X0 = None,
        iters = None,
        verbose = False, 
        verbose_indent = 4,
        quantiles_losses = [0,1,5,10,25,40,50,60,75,90,95,99,100],
        verbose_quantiles_losses = [5,25,50,75,90],
        verbose_times = True, 
        warn = True,
        return_data = False,
        ):
    r"""
    [MINRES algorithm](https://en.wikipedia.org/wiki/Minimal_residual_method) for solving symmetric linear systems 

    $$AX=B$$

    A translation of `[scipy.sparse.linalg.minres](https://github.com/scipy/scipy/blob/v1.17.0/scipy/sparse/linalg/_isolve/minres.py)`.

    Args:
        A (Union[torch.Tensor,callable]): Symmetric matrix `A` with shape `(...,n,n)`, or  
            `callable(A)` where `a(x)` should return the batch matrix multiplication of `A` and `X`,  
        B (torch.Tensor): Right hand side tensor $B$ with shape `(...,n,k)`
        X0 (torch.Tensor): Initial guess for $X$ with shape `(...,n,k)`, defaults to zeros. 
        iters (int): number of minres iterations, defaults to `5n`. 
        verbose (int): Controls logging verbosity

            - If True, perform logging. 
            - If a positive int, only log every verbose iterations. 
            - If None, set to a reasonable positive int based on the maximum number of iterations
            - If False, don't log. 
        
        verbose_indent (int): Positive number of indentation spaces for logging.
        quantiles_losses (list): Loss quantiles to record.
        verbose_quantiles_losses (list): Loss quantiles to show in verbose log.
        verbose_times (bool): If `False`, do not show the times in the verbose log. This is mostly for testing where timing is not reproducible. 
        warn (bool): If `False`, then suppress warnings.
        return_data (bool): If `True`, return `(x,data)`, otherwise only return `x`
    
    Returns:
        x (torch.Tensor): Optimized $X$.
        data (dict): Iteration data, only returned when `return_data=True`

    Examples:

        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

        Column vector $b$ 
        
        >>> n = 5
        >>> A = torch.randn(n,n,generator=rng)
        >>> A = (A+A.T)/2
        >>> b = torch.rand(n,generator=rng)
        >>> x_true = torch.linalg.solve(A,b[...,None])[...,0]
        >>> x_true
        tensor([-0.1402,  0.4565,  0.2920,  0.2470,  0.3251])
        >>> torch.allclose(A@x_true-b,torch.zeros_like(b))
        True
        >>> x_minres = minres(A,b[...,None],verbose=None,verbose_times=False)[...,0]
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 6.1e-01   | 6.1e-01   | 6.1e-01   | 6.1e-01   | 6.1e-01   
            2          | 2.1e-01   | 2.1e-01   | 2.1e-01   | 2.1e-01   | 2.1e-01   
            3          | 1.9e-01   | 1.9e-01   | 1.9e-01   | 1.9e-01   | 1.9e-01   
            4          | 7.3e-02   | 7.3e-02   | 7.3e-02   | 7.3e-02   | 7.3e-02   
            5          | 5.4e-16   | 5.4e-16   | 5.4e-16   | 5.4e-16   | 5.4e-16   
        >>> torch.allclose(x_minres,x_true)
        True

        Matrix $B$
        
        >>> n = 5
        >>> k = 3
        >>> A = torch.randn(n,n,generator=rng)
        >>> A = (A+A.T)/2
        >>> B = torch.rand(n,k,generator=rng)
        >>> X_true = torch.linalg.solve(A,B)
        >>> X_true
        tensor([[ 4.3521,  3.1162,  2.4974],
                [ 5.2334,  3.3709,  2.8593],
                [ 2.2800,  1.8017,  1.2739],
                [ 0.1085, -0.1715,  0.0143],
                [ 1.3653,  1.5421,  1.0192]])
        >>> torch.allclose(A@X_true-B,torch.zeros_like(B))
        True
        >>> X_minres = minres(A,B,verbose=None,verbose_times=False)
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 9.0e-01   | 9.1e-01   | 9.1e-01   | 9.5e-01   | 9.8e-01   
            2          | 6.5e-01   | 6.8e-01   | 7.2e-01   | 7.6e-01   | 7.7e-01   
            3          | 4.8e-01   | 4.8e-01   | 4.8e-01   | 5.8e-01   | 6.4e-01   
            4          | 4.4e-01   | 4.5e-01   | 4.6e-01   | 5.6e-01   | 6.3e-01   
            5          | 4.4e-15   | 5.8e-15   | 7.7e-15   | 7.9e-15   | 8.0e-15   
        >>> torch.allclose(X_minres,X_true)
        True

        Tri-diagonal $A$ with torage-saving multiplication function 
        
        >>> n = 5
        >>> k = 3
        >>> A_diag = torch.randn(n,generator=rng)
        >>> A_off_diag = torch.randn(n-1,generator=rng) 
        >>> A = torch.zeros(n,n)
        >>> A[torch.arange(n),torch.arange(n)] = A_diag 
        >>> A[torch.arange(n-1),torch.arange(1,n)] = A_off_diag
        >>> A[torch.arange(1,n),torch.arange(n-1)] = A_off_diag
        >>> A
        tensor([[-1.0765, -0.8797,  0.0000,  0.0000,  0.0000],
                [-0.8797, -2.1098, -1.0459,  0.0000,  0.0000],
                [ 0.0000, -1.0459, -0.8007, -1.1058,  0.0000],
                [ 0.0000,  0.0000, -1.1058, -0.0095, -1.4746],
                [ 0.0000,  0.0000,  0.0000, -1.4746,  0.8703]])
        >>> B = torch.rand(n,k,generator=rng)
        >>> X_true = torch.linalg.solve(A,B)
        >>> X_true
        tensor([[ 0.1212, -1.2537, -0.4189],
                [-1.1412,  0.7219, -0.2515],
                [ 1.4109, -0.8289, -0.0219],
                [-0.8073, -0.3599, -0.5720],
                [-1.2689,  0.4987,  0.0151]])
        >>> torch.allclose(A@X_true-B,torch.zeros_like(B))
        True
        >>> def A_mult(x):
        ...     y = x*A_diag[:,None]
        ...     y[1:,:] += x[:-1,:]*A_off_diag[:,None]
        ...     y[:-1,:] += x[1:,:]*A_off_diag[:,None]
        ...     return y
        >>> torch.allclose(A_mult(X_true),A@X_true)
        True
        >>> X_minres,data = minres(A_mult,B,verbose=None,verbose_times=False,return_data=True)
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 4.0e-01   | 5.6e-01   | 7.7e-01   | 8.4e-01   | 8.8e-01   
            2          | 2.2e-01   | 2.8e-01   | 3.5e-01   | 4.7e-01   | 5.4e-01   
            3          | 1.1e-01   | 1.9e-01   | 2.9e-01   | 3.1e-01   | 3.2e-01   
            4          | 4.0e-02   | 1.5e-01   | 2.8e-01   | 2.8e-01   | 2.8e-01   
            5          | 7.6e-15   | 8.4e-15   | 9.5e-15   | 9.9e-15   | 1.0e-14   
        >>> torch.allclose(X_minres,X_true)
        True
        >>> 

        Batched tri-diagonal $A$ with torage-saving multiplication function 

        >>> n = 100
        >>> k = 3
        >>> A_diag = torch.randn(2,1,4,n,generator=rng)
        >>> A_off_diag = torch.randn(2,1,4,n-1,generator=rng) 
        >>> A = torch.zeros(2,1,4,n,n)
        >>> A[...,torch.arange(n),torch.arange(n)] = A_diag 
        >>> A[...,torch.arange(n-1),torch.arange(1,n)] = A_off_diag
        >>> A[...,torch.arange(1,n),torch.arange(n-1)] = A_off_diag
        >>> B = torch.rand(2,6,1,n,k,generator=rng)
        >>> X_true = torch.linalg.solve(A,B)
        >>> torch.allclose(torch.einsum("...ij,...jk->...ik",A,X_true)-B,torch.zeros_like(B))
        True
        >>> def A_mult(x):
        ...     y = x*A_diag[...,:,None]
        ...     y[...,1:,:] += x[...,:-1,:]*A_off_diag[...,:,None]
        ...     y[...,:-1,:] += x[...,1:,:]*A_off_diag[...,:,None]
        ...     return y
        >>> torch.allclose(A_mult(X_true),torch.einsum("...ij,...jk->...ik",A,X_true))
        True
        >>> X_minres = minres(A_mult,B,verbose=None,verbose_times=False)
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            25         | 1.9e-01   | 2.3e-01   | 2.5e-01   | 2.8e-01   | 3.1e-01   
            50         | 7.4e-02   | 1.2e-01   | 1.4e-01   | 1.6e-01   | 2.0e-01   
            75         | 2.1e-02   | 4.0e-02   | 7.9e-02   | 1.1e-01   | 1.5e-01   
            100        | 3.1e-03   | 7.2e-03   | 2.6e-02   | 7.6e-02   | 1.2e-01   
            125        | 3.3e-07   | 2.9e-06   | 2.1e-04   | 5.6e-03   | 7.9e-02   
            150        | 2.1e-15   | 1.0e-14   | 1.7e-12   | 1.3e-10   | 2.0e-09   
            151        | 2.1e-15   | 8.7e-15   | 1.1e-12   | 8.9e-11   | 1.2e-09   
        >>> X_minres.shape
        torch.Size([2, 6, 4, 100, 3])
        >>> torch.allclose(X_minres,X_true)
        True
        >>> print_data_signatures(data)
            data['x'].shape = (5, 3)
            data['iterrange'].shape = (26,)
            data['times'].shape = (26,)
            data['xs'].shape = (26, 5, 3)
            data['losses'].shape = (26, 3)
            data['losses_quantiles']
                data['losses_quantiles']['0'].shape = (26,)
                data['losses_quantiles']['1'].shape = (26,)
                data['losses_quantiles']['5'].shape = (26,)
                data['losses_quantiles']['10'].shape = (26,)
                data['losses_quantiles']['25'].shape = (26,)
                data['losses_quantiles']['40'].shape = (26,)
                data['losses_quantiles']['50'].shape = (26,)
                data['losses_quantiles']['60'].shape = (26,)
                data['losses_quantiles']['75'].shape = (26,)
                data['losses_quantiles']['90'].shape = (26,)
                data['losses_quantiles']['95'].shape = (26,)
                data['losses_quantiles']['99'].shape = (26,)
                data['losses_quantiles']['100'].shape = (26,)
    """
    if warn and (not torch.get_default_dtype()==torch.float64): warnings.warn('''
            torch.get_default_dtype() = %s, but lm_opt often requires high precision updates. We recommend using:
                torch.set_default_dtype(torch.float64)'''%str(torch.get_default_dtype()))
    device = str(B.device)
    default_device = str(torch.get_default_device())
    assert B.ndim>=2, "B should have shape (...,n,k)"
    n = B.size(-2)
    k = B.size(-1)
    if X0 is None: 
        X0 = torch.zeros_like(B)
    if isinstance(A,torch.Tensor):
        assert A.shape[-2:]==(n,n)
        assert torch.allclose(A.transpose(dim0=-2,dim1=-1),A)
        matvec = lambda X: torch.einsum("...ij,...jk->...ik",A,X)
    else:
        assert callable(A)
        matvec = A
    if iters is None: 
        iters = 5*n 
    assert X0.shape==B.shape 
    assert isinstance(return_data,bool)
    assert iters>=0
    assert iters%1==0
    if verbose is None: 
        verbose = max(1,iters//20)
    assert isinstance(quantiles_losses,list)
    assert all(0<=qt<=100 for qt in quantiles_losses)
    assert isinstance(verbose_quantiles_losses,list)
    assert all(qt in quantiles_losses for qt in verbose_quantiles_losses)
    assert verbose%1==0
    assert verbose>=0 
    assert verbose_indent%1==0 
    assert verbose_indent>=0
    assert isinstance(verbose_times,bool)
    if verbose:
        _h_iter = "%-10s "%"iter i"
        _h_times = "| %-10s"%"times" if verbose_times else ""
        _s_losses_qt = ("| %-9s "*len(verbose_quantiles_losses))%tuple(str(qt) for qt in verbose_quantiles_losses)
        _h_losses_qt = "| losses_quantiles"+" "*(len(_s_losses_qt)-len("| losses_quantiles"))
        _h = _h_iter+_h_losses_qt+_h_times
        _s = " "*len(_h_iter)+_s_losses_qt+("|"+" "*(len(_h_times)-1) if verbose_times else " "*len(_h_times))
        print(" "*verbose_indent+_h)
        print(" "*verbose_indent+_s)
        print(" "*verbose_indent+"~"*len(_s))
    timer = Timer(device=device)
    timer.tic()
    psolve = lambda X: X # TODO: implement more involved preconditioned solver
    inner = lambda a,b: torch.einsum("...ij,...ij->...j",a,b)
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0
    eps = torch.finfo(B.dtype).eps
    x = X0 
    Ax = matvec(x)
    assert Ax.shape[-2:]==(n,k)
    batch_shape = tuple(Ax.shape[:-2])
    xs = torch.nan*torch.ones((iters+1,*batch_shape,n,k),device=default_device)
    times = torch.nan*torch.ones(iters+1,device=default_device)
    losses = torch.nan*torch.ones((iters+1,*batch_shape,k),device=default_device)
    losses_quantiles = {str(qt):torch.nan*torch.ones(iters+1,device=default_device) for qt in quantiles_losses}
    r1 = B-Ax # (...,n,k)
    y = psolve(r1) # (...,n,k)
    beta1 = inner(r1,y) # (...,k)
    if (beta1<0).any():
        raise ValueError('indefinite preconditioner')
    bnorm = torch.linalg.norm(B,dim=-2) # (...,k)
    beta1 = torch.sqrt(beta1)
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = torch.zeros(1)
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    gmax = torch.zeros(1)
    gmin = torch.finfo(B.dtype).max*torch.ones(1)
    cs = -1
    sn = 0
    w = torch.zeros_like(B)
    w2 = torch.zeros_like(B)
    r2 = r1
    shift = 0 # TODO: If shift != 0 then the method solves (A - shift*I)x = b
    rtol = 1e-5 # TODO: 
    residtol = 1e-8 # TODO: 
    for i in range(iters+1):
        xs[i] = x.to(default_device)
        r = matvec(x)-B 
        loss = torch.linalg.norm(r,dim=-2)/bnorm
        losses[i] = loss.to(default_device)
        for qt in quantiles_losses:
            losses_quantiles[str(qt)][i] = loss.nanquantile(qt/100).to(default_device)
        times[i] = timer.toc()
        breakcond = i==iters or r.abs().amax()<=residtol
        if verbose and (i%verbose==0 or breakcond):
            _s_iter = "%-10d "%i
            _s_losses_qt = ("| %-9.1e "*len(verbose_quantiles_losses))%tuple(losses_quantiles[str(qt)][i] for qt in verbose_quantiles_losses)
            _s_times = "| %-10.1f "%(times[i]) if verbose_times else ""
            print(" "*verbose_indent+_s_iter+_s_losses_qt+_s_times)
        if breakcond: break 
        s = 1/beta
        v = s[...,None,:]*y
        y = matvec(v)
        y = y-shift*v
        if i>0:
            y = y-(beta/oldb)[...,None,:]*r1
        alfa = inner(v,y)
        y = y-(alfa/beta)[...,None,:]*r2
        r1 = r2
        r2 = y
        y = psolve(r2)
        oldb = beta
        beta = inner(r2,y)
        if (beta<0).any():
            raise ValueError('non-symmetric matrix')
        beta = torch.sqrt(beta)
        tnorm2 += alfa**2+oldb**2+beta**2
        oldeps = epsln
        delta = cs*dbar+sn*alfa
        gbar = sn*dbar-cs*alfa
        epsln = sn*beta
        dbar = -cs*beta
        root = torch.linalg.norm(torch.stack([gbar,dbar],dim=-1),dim=-1)
        Arnorm = phibar*root
        gamma = torch.linalg.norm(torch.stack([gbar,beta],dim=-1),dim=-1)
        gamma = torch.maximum(gamma,eps*torch.ones(1))
        cs = gbar/gamma
        sn = beta/gamma
        phi = cs*phibar
        phibar = sn*phibar
        denom = 1/gamma
        w1 = w2
        w2 = w
        w = (v-oldeps[...,None,:]*w1-delta[...,None,:]*w2)*denom[...,None,:]
        x = x+phi[...,None,:]*w
        gmax = torch.maximum(gmax,gamma)
        gmin = torch.minimum(gmin,gamma)
        z = rhs1/gamma
        rhs1 = rhs2-delta*z
        rhs2 = -epsln*z
        Anorm = torch.sqrt(tnorm2)
        ynorm = torch.linalg.norm(x,dim=-2)
        epsa = Anorm*eps
        epsx = Anorm*ynorm*eps
        epsr = Anorm*ynorm*rtol
        diag = gbar
        diag = torch.where(diag==0,epsa,diag)
        qrnorm = phibar
        rnorm = qrnorm
        Acond = gmax/gmin
    if not return_data:
        return x 
    else:
        data = {
            "x": x.to(default_device), 
            "iterrange": torch.arange(iters+1), 
            "times": times, 
            "xs": xs,
            "losses": losses,
            "losses_quantiles": losses_quantiles,
            }
        return x,data

if __name__=="__main__":
    device = "cpu"
    if "mps" not in device:
        torch.set_default_dtype(torch.float64)
    rng = torch.Generator(device=device).manual_seed(7)

    # x = torch.rand((10,4,),generator=rng,device=device)
    # theta_true = torch.rand((4,),generator=rng,device=device)
    # ytrue = torch.exp((x*theta_true).sum(-1)) # (10,)
    # def f(theta):
    #     yhat = torch.exp((x*theta[...,None,:]).sum(-1)) # (...,10)
    #     return yhat
    # theta,data = lm_opt(
    #     f = f, 
    #     theta0 = torch.rand_like(theta_true,generator=rng),
    #     ytrue = ytrue,
    #     iters = 3,
    #     # jacfwd=False,
    #     )
    # print_data_signatures(data,show_device=True)

    # x = torch.rand((3,3,3,2,2),generator=rng,device=device)
    # theta_true = torch.rand((4,4,2,2),generator=rng,device=device)
    # ytrue = torch.exp((x*theta_true[...,None,None,None,:,:]).sum((-2,-1))) # (4,4,3,3,3)
    # def f(theta):
    #     yhat = torch.exp((x*theta[...,None,None,None,:,:]).sum((-2,-1))) # (...,3,3,3)
    #     return yhat
    # theta_hat,data = lm_opt(
    #     f = f, 
    #     theta0 = torch.rand_like(theta_true,generator=rng),
    #     ytrue = ytrue,
    #     iters = 20,
    #     batch_dims = 2,
    #     lam_factors = [torch.tensor([1/4,1/2,1,2,4])],
    #     alpha_factors = [torch.tensor([2/3,1,3/2])],
    #     verbose_times = False,
    #     )
    # print_data_signatures(data)

    n = 500
    k = 3
    A_diag = 2+torch.randn(2,4,n,generator=rng)
    A_off_diag = torch.randn(2,4,n-1,generator=rng) 
    A = torch.zeros(2,4,n,n)
    A[...,torch.arange(n),torch.arange(n)] = A_diag 
    A[...,torch.arange(n-1),torch.arange(1,n)] = A_off_diag
    A[...,torch.arange(1,n),torch.arange(n-1)] = A_off_diag
    print(torch.linalg.cond(A))
    B = torch.rand(2,4,n,k,generator=rng)
    X_true = torch.linalg.solve(A,B)
    torch.allclose(torch.einsum("...ij,...jk->...ik",A,X_true)-B,torch.zeros_like(B))
    def A_mult(x):
        y = x*A_diag[...,:,None]
        y[...,1:,:] += x[...,:-1,:]*A_off_diag[...,:,None]
        y[...,:-1,:] += x[...,1:,:]*A_off_diag[...,:,None]
        return y
    torch.allclose(A_mult(X_true),torch.einsum("...ij,...jk->...ik",A,X_true))
    X_minres = minres(A_mult,B,iters=1000)
    print(X_minres)

    # def A_mult(x):
    #     y = x*A_diag[...,:,None]
    #     y[...,1:,:] += x[...,:-1,:]*A_off_diag[...,:,None]
    #     y[...,:-1,:] += x[...,1:,:]*A_off_diag[...,:,None]
    #     return y[1,1,:,[0]]
    # X_minres = minres(A_mult,B[1,1,:,[0]],iters=1000)
    
    
    
    # import scipy.sparse
    # x,info = scipy.sparse.linalg.minres(A[1,1,:,:].numpy(),B[1,1,:,0].numpy(),rtol=1e-16)
    # print(x.shape)
    # r = A[1,1,:,:].numpy()@x-B[1,1,:,0].numpy()
    # print((r**2).sum())
    # print(info)
