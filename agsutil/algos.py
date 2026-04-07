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

def print_data_signatures_lm_opt(data, show_device=False):
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
        verbose = None, 
        verbose_indent = 4,
        quantiles_losses = [0,1,5,10,25,40,50,60,75,90,95,99,100],
        quantiles_lams =   [0,1,5,10,25,40,50,60,75,90,95,99,100],
        quantiles_alphas = [0,1,5,10,25,40,50,60,75,90,95,99,100],
        verbose_quantiles_losses = [5,25,50,75,90],
        verbose_quantiles_lams =   [5,25,50,75,90],
        verbose_quantiles_alphas = [5,25,50,75,90],
        warn = True,
        ):
    r"""
    Levenberg--Marquardt optimization 

    Args:
        f (func): Residual function. 
        theta0 (torch.Tensor): Initial guess for parameters $\theta$. 
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
            - If False, don't log. 
        
        verbose_indent (int): Positive number of indentation spaces for logging.
        quantiles_losses (list): Loss quantiles to record.
        quantiles_lams (list): $\lambda$ quantiles to record.
        quantiles_alphas (list): $\alpha$ quantiles to record.
        verbose_quantiles_losses (list): Loss quantiles to show in verbose log.
        verbose_quantiles_lams (list): $\lambda$ quantiles to show in verbose log.
        verbose_quantiles_alphas (list): $\alpha$ quantiles to show in verbose log.
        warn (bool): If `False`, then suppress warnings.
    
    Returns:
        theta (torch.Tensor): Optimized parameters.
        data (dict): Iteration data.

    Examples:

        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)
        >>> x = torch.rand((3,3,3,2,2),generator=rng)
        >>> theta_true = torch.rand((4,4,2,2),generator=rng)
        >>> y_true = torch.exp((x*theta_true[...,None,None,None,:,:]).sum((-2,-1))) # (4,4,3,3,3)
        >>> def f(theta,y_true):
        ...     y_hat = torch.exp((x*theta[...,None,None,None,:,:]).sum((-2,-1))) # (...,3,3,3)
        ...     return (y_hat-y_true),y_true
        >>> theta_hat,data = lm_opt(
        ...     f = f, 
        ...     theta0 = torch.rand_like(theta_true,generator=rng),
        ...     iters = 5,
        ...     batch_dims = 2,
        ...     f_kwargs_vec = {"y_true":y_true},
        ...     f_kwargs_no_vec = {},
        ...     lam_factors = [torch.tensor([1/4,1/2,1,2,4])],
        ...     alpha_factors = [torch.tensor([2/3,1,3/2])],
        ...     )
            iter i     | losses_quantiles                                          | lams_quantiles                                            | alphas_quantiles                                          | times     
                       | 5         | 25        | 50        | 75        | 90        | 5         | 25        | 50        | 75        | 90        | 5         | 25        | 50        | 75        | 90                    
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.5e+01   | 2.5e+01   | 6.3e+01   | 1.9e+02   | 3.1e+02   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | ...        
            1          | 2.2e-01   | 6.0e-01   | 1.0e+00   | 1.7e+00   | 2.0e+01   | 2.5e-07   | 2.5e-07   | 1.0e-06   | 1.0e-06   | 3.7e-06   | 6.7e-01   | 9.2e-01   | 1.0e+00   | 1.0e+00   | 1.5e+00   | ...        
            2          | 1.4e-04   | 3.0e-04   | 1.6e-03   | 2.9e-03   | 2.0e-01   | 6.2e-08   | 2.5e-07   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 6.7e-01   | 6.7e-01   | 1.0e+00   | 1.0e+00   | 1.5e+00   | ...        
            3          | 4.4e-11   | 3.2e-10   | 3.9e-09   | 7.5e-09   | 4.5e-05   | 1.6e-08   | 2.5e-07   | 2.5e-07   | 1.0e-06   | 1.0e-06   | 6.7e-01   | 6.7e-01   | 1.0e+00   | 1.0e+00   | 1.5e+00   | ...        
            4          | 5.1e-24   | 6.4e-22   | 1.6e-20   | 9.6e-20   | 3.1e-12   | 7.6e-09   | 6.2e-08   | 2.5e-07   | 1.0e-06   | 1.0e-06   | 6.7e-01   | 6.7e-01   | 1.0e+00   | 1.0e+00   | 1.5e+00   | ...        
            5          | 1.0e-30   | 1.6e-30   | 3.8e-30   | 1.3e-29   | 1.6e-26   | 3.4e-09   | 1.6e-08   | 2.5e-07   | 1.0e-06   | 1.0e-06   | 6.7e-01   | 6.7e-01   | 1.0e+00   | 1.0e+00   | 1.5e+00   | ...        
        >>> print_data_signatures_lm_opt(data)
            data['iterrange'].shape = (6,)
            data['times'].shape = (6,)
            data['theta'].shape = (4, 4, 2, 2)
            data['thetas'].shape = (6, 4, 4, 2, 2)
            data['losses'].shape = (6, 4, 4)
            data['lams'].shape = (6, 4, 4)
            data['alphas'].shape = (6, 4, 4)
            data['losses_quantiles']
                data['losses_quantiles']['0'].shape = (6,)
                data['losses_quantiles']['1'].shape = (6,)
                data['losses_quantiles']['5'].shape = (6,)
                data['losses_quantiles']['10'].shape = (6,)
                data['losses_quantiles']['25'].shape = (6,)
                data['losses_quantiles']['40'].shape = (6,)
                data['losses_quantiles']['50'].shape = (6,)
                data['losses_quantiles']['60'].shape = (6,)
                data['losses_quantiles']['75'].shape = (6,)
                data['losses_quantiles']['90'].shape = (6,)
                data['losses_quantiles']['95'].shape = (6,)
                data['losses_quantiles']['99'].shape = (6,)
                data['losses_quantiles']['100'].shape = (6,)
            data['lams_quantiles']
                data['lams_quantiles']['0'].shape = (6,)
                data['lams_quantiles']['1'].shape = (6,)
                data['lams_quantiles']['5'].shape = (6,)
                data['lams_quantiles']['10'].shape = (6,)
                data['lams_quantiles']['25'].shape = (6,)
                data['lams_quantiles']['40'].shape = (6,)
                data['lams_quantiles']['50'].shape = (6,)
                data['lams_quantiles']['60'].shape = (6,)
                data['lams_quantiles']['75'].shape = (6,)
                data['lams_quantiles']['90'].shape = (6,)
                data['lams_quantiles']['95'].shape = (6,)
                data['lams_quantiles']['99'].shape = (6,)
                data['lams_quantiles']['100'].shape = (6,)
            data['alphas_quantiles']
                data['alphas_quantiles']['0'].shape = (6,)
                data['alphas_quantiles']['1'].shape = (6,)
                data['alphas_quantiles']['5'].shape = (6,)
                data['alphas_quantiles']['10'].shape = (6,)
                data['alphas_quantiles']['25'].shape = (6,)
                data['alphas_quantiles']['40'].shape = (6,)
                data['alphas_quantiles']['50'].shape = (6,)
                data['alphas_quantiles']['60'].shape = (6,)
                data['alphas_quantiles']['75'].shape = (6,)
                data['alphas_quantiles']['90'].shape = (6,)
                data['alphas_quantiles']['95'].shape = (6,)
                data['alphas_quantiles']['99'].shape = (6,)
                data['alphas_quantiles']['100'].shape = (6,)
        >>> torch.allclose(theta_hat,theta_true)
        True
    """
    if warn and (not torch.get_default_dtype()==torch.float64):
        warnings.warn('''
            torch.get_default_dtype() = %s, but lm_opt often requires high precision updates. We recommend using:
                torch.set_default_dtype(torch.float64)
            '''%str(torch.get_default_dtype()))
    assert callable(f) 
    assert batch_dims>=0
    assert isinstance(theta0,torch.Tensor)
    batch_shape = tuple(theta0.shape[:batch_dims])
    R = int(torch.tensor(batch_shape).prod())
    nonbatch_theta_dims = theta0.ndim-batch_dims
    nonbatch_theta_shape = tuple(theta0.shape[batch_dims:])
    T = int(torch.tensor(nonbatch_theta_shape).prod())
    device = str(theta0.device)
    default_device = str(torch.get_default_device())
    assert iters%1==0, "iters should be an int"
    assert iters>=0
    if batch_dims==0:
        theta = theta0[None,...]
    else: # batch_dims>0:
        theta = theta0.flatten(end_dim=batch_dims-1)
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
    thetas = torch.nan*torch.ones((iters+1,*batch_shape,*nonbatch_theta_shape),device=default_device)
    times = torch.nan*torch.ones(iters+1,device=default_device)
    losses = torch.nan*torch.ones((iters+1,*batch_shape),device=default_device)
    lams = torch.nan*torch.ones((iters+1,*batch_shape),device=default_device)
    alphas = torch.nan*torch.ones((iters+1,*batch_shape),device=default_device)
    losses_quantiles = {str(qt):torch.nan*torch.ones(iters+1,device=default_device) for qt in quantiles_losses}
    lams_quantiles = {str(qt):torch.nan*torch.ones(iters+1,device=default_device) for qt in quantiles_lams}
    alphas_quantiles = {str(qt):torch.nan*torch.ones(iters+1,device=default_device) for qt in quantiles_alphas}
    if verbose:
        _h_iter = "%-10s "%"iter i"
        _h_times = "| %-10s"%"times"
        _s_losses_qt = ("| %-9s "*len(verbose_quantiles_losses))%tuple(str(qt) for qt in verbose_quantiles_losses)
        _s_lams_qt = ("| %-9s "*len(verbose_quantiles_lams))%tuple(str(qt) for qt in verbose_quantiles_lams)
        _s_alphas_qt = ("| %-9s "*len(verbose_quantiles_alphas))%tuple(str(qt) for qt in verbose_quantiles_alphas)
        _h_losses_qt = "| losses_quantiles"+" "*(len(_s_losses_qt)-len("| losses_quantiles"))
        _h_lams_qt   = "| lams_quantiles"  +" "*(len(_s_lams_qt)  -len("| lams_quantiles"))
        _h_alphas_qt = "| alphas_quantiles"+" "*(len(_s_alphas_qt)-len("| alphas_quantiles"))
        _h = _h_iter+_h_losses_qt+_h_lams_qt+_h_alphas_qt+_h_times
        _s = " "*len(_h_iter)+_s_losses_qt+_s_lams_qt+_s_alphas_qt+" "*len(_h_times)
        print(" "*verbose_indent+_h)
        print(" "*verbose_indent+_s)
        print(" "*verbose_indent+"~"*len(_s))
    def ftilde(theta, *f_kwargs_vec_vals):
        assert len(f_kwargs_vec_vals)==len(f_kwargs_vec_names)
        f_kwargs_vec = {f_kwargs_vec_names[i]:f_kwargs_vec_vals[i] for i in range(len(f_kwargs_vec_names))}
        y,*others = f(theta,**f_kwargs_vec,**f_kwargs_no_vec)
        return y,(y,*others)
    assert isinstance(jacfwd,bool)
    if jacfwd:
        jac_ftilde = torch.func.jacfwd(ftilde,argnums=(0,),has_aux=True)
    else:
        jac_ftilde = torch.func.jacrev(ftilde,argnums=(0,),has_aux=True)
    vjac_ftilde = torch.func.vmap(jac_ftilde,in_dims=(0,)+(0,)*len(f_kwargs_vec_names),chunk_size=vmap_chunk_size)
    eyeT = torch.eye(T,device=device)
    Rrange = torch.arange(R,device=device)
    lam = lam0*torch.ones(R,device=device)
    alpha = alpha0*torch.ones(R,device=device)
    timer = Timer(device=device)
    timer.tic()
    for i in range(iters+1):
        if i==iters:
            _,(resid,*others) = ftilde(theta,*f_kwargs_vec_vals)
        else:
            (Jfull,),(resid,*others) = vjac_ftilde(theta,*f_kwargs_vec_vals)
        nonbatch_resid_dims = resid.ndim-1
        nonbatch_resid_shape = tuple(resid.shape[1:])
        K = int(torch.tensor(nonbatch_resid_shape).prod())
        if i==0 and warn:
            if jacfwd and T>K:
                warnings.warn('''
                For T the number of inputs and K the number of outputs:
                    torch.func.jacfwd performs best when T << K. 
                    torch.func.jacrev performs best when K << T.
                You are using torch.func.jacfwd but T = %d > %d = K. 
                Try using torch.func.jacrev by setting jacfwd = False.
                '''%(T,K))
            if (not jacfwd) and T<K:
                warnings.warn('''
                For T the number of inputs and K the number of outputs:
                    torch.func.jacfwd performs best when T << K. 
                    torch.func.jacrev performs best when K << T.
                You are using torch.func.jacrev but T = %d < %d = K. 
                Try using torch.func.jacrev by setting jacfwd = True.
                '''%(T,K))
        assert Jfull.shape==(R,*nonbatch_resid_shape,*nonbatch_theta_shape)
        loss = (resid**2).flatten(start_dim=1).sum(-1)
        losses[i] = loss.reshape(batch_shape).to(default_device)
        lams[i] = lam.reshape(batch_shape).to(default_device)
        alphas[i] = alpha.reshape(batch_shape).to(default_device)
        for qt in quantiles_losses:
            losses_quantiles[str(qt)][i] = loss.nanquantile(qt/100).to(default_device)
        for qt in quantiles_lams:
            lams_quantiles[str(qt)][i] = lams.nanquantile(qt/100).to(default_device)
        for qt in quantiles_alphas:
            alphas_quantiles[str(qt)][i] = alphas.nanquantile(qt/100).to(default_device)
        times[i] = timer.toc()
        if verbose and (i%verbose==0 or i==iters):
            _s_iter = "%-10d "%i
            _s_losses_qt = ("| %-9.1e "*len(verbose_quantiles_losses))%tuple(losses_quantiles[str(qt)][i] for qt in verbose_quantiles_losses)
            _s_lams_qt = ("| %-9.1e "*len(verbose_quantiles_lams))%tuple(lams_quantiles[str(qt)][i] for qt in verbose_quantiles_lams)
            _s_alphas_qt = ("| %-9.1e "*len(verbose_quantiles_alphas))%tuple(alphas_quantiles[str(qt)][i] for qt in verbose_quantiles_alphas)
            _s_times = "| %-10.1f "%(times[i])
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
        _,(resid_new_success,*others_new_success) = ftilde(thetas_new[:,success],*f_kwargs_vec_vals_success)
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
        "iterrange": torch.arange(iters+1), 
        "times": times, 
        "theta": theta.to(default_device), 
        "thetas": thetas,
        "losses": losses,
        "lams": lams, 
        "alphas": alphas,
        "losses_quantiles": losses_quantiles,
        "lams_quantiles": lams_quantiles,
        "alphas_quantiles": alphas_quantiles,
        }
    return theta,data

if __name__=="__main__":
    device = "cpu"
    if "mps" not in device:
        torch.set_default_dtype(torch.float64)
    rng = torch.Generator(device=device).manual_seed(7)
    x = torch.rand((10,4,),generator=rng,device=device)
    theta_true = torch.rand((4,),generator=rng,device=device)
    y_true = torch.exp((x*theta_true).sum(-1)) # (10,)
    def f(theta,y_true):
        y_hat = torch.exp((x*theta[...,None,:]).sum(-1)) # (...,10)
        return (y_hat-y_true),y_true
    theta,data = lm_opt(
        f = f, 
        theta0 = torch.rand_like(theta_true,generator=rng),
        iters = 5,
        batch_dims = 0,
        f_kwargs_vec = {"y_true":y_true},
        f_kwargs_no_vec = {},
        # lam_factors = [torch.tensor([1/2,1,2]),torch.tensor([1])],
        # alpha_factors = [torch.tensor([1]),torch.tensor([1/2,1,2])],
        )
    print_data_signatures_lm_opt(data,show_device=True)
    print(theta.device)
