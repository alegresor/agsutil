import torch 
import numpy as np
import warnings 
from .utils import Timer,print_data_signatures

def lm_opt(
        f,
        theta0,
        ytrue,
        batch_dims = 0, 
        iters = 10,
        residtol = None,
        minimize = True,
        loss_mult = 1,
        loss_shift = 0,
        f_kwargs_vec = {},
        f_kwargs_no_vec = {},
        lam0 = 1e-6,
        alpha0 = 1e0,
        lam_factors = [[1/2,1,2]],
        alpha_factors = [[1/2,1,2]],
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
        store_data_iters = False,
        store_all_data = False, 
        ):
    r"""
    Levenberg--Marquardt optimization 

    Args:
        f (func): Residual function. 
        theta0 (torch.Tensor): Initial guess for parameters $\theta$. 
        ytrue (torch.Tensor): True `y` values, i.e. `f(theta_true)`. 
        batch_dims (int): Number of batch dimension. 
        iters (int): Number of iterations. 
        residtol (float): Non-negative tolerance on the maximum residual for early stopping, defaults to `1e-12` for `torch.float64` and `2.5e-4` for `torch.float32`.
        minimize (bool): If `True`, minimize the objective, otherwise maximize the objective. 
        loss_mult (bool): Scalar amount by which to multiply the loss so `loss = loss_mult*torch.sum(resid**2,dim=-1)+loss_shift`.
        loss_shift (bool): Scalar amount by which to shift the loss so `loss = loss_mult*torch.sum(resid**2,dim=-1)+loss_shift`.
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
        
            - If `True`, perform logging. 
            - If a positive int, only log every verbose iterations. 
            - If `None`, set to a reasonable positive int based on the maximum number of iterations
            - If `False`, don't log. 
        
        verbose_indent (int): Non-negative number of indentation spaces for logging.
        quantiles_losses (list): Loss quantiles to record.
        quantiles_lams (list): $\lambda$ quantiles to record.
        quantiles_alphas (list): $\alpha$ quantiles to record.
        verbose_quantiles_losses (list): Loss quantiles to show in verbose log.
        verbose_quantiles_lams (list): $\lambda$ quantiles to show in verbose log.
        verbose_quantiles_alphas (list): $\alpha$ quantiles to show in verbose log.
        verbose_times (bool): If `False`, do not show the times in the verbose log. This is mostly for testing where timing is not reproducible. 
        warn (bool): If `False`, then suppress warnings.
        store_data_iters (int): Controls storage iterations with the same options as verbose. If `store_data_iters==0`, then the data is not collected or returned. 

            - If `True`, store every iteration. 
            - If a positive int, only store every `store_data_iters` iterations. 
            - If `None`, set to a reasonable positive int based on the maximum number of iterations
            - If `False`, don't store data, and do not return data 

        store_all_data (bool): If `True`, store the `x` values as well as the metrics. 
    
    Returns:
        theta (torch.Tensor): Optimized parameters.
        data (dict): Iteration data, only returned when `store_data_iters>0`

    Examples:

        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

        
    Standard example
    
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
        ...     store_data_iters = None,
        ...     store_all_data = True,
        ...     )
            iter i     | losses_quantiles                                          | lams_quantiles                                            | alphas_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        | 5         | 25        | 50        | 75        | 90        | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 2.3e+01   | 2.3e+01   | 2.3e+01   | 2.3e+01   | 2.3e+01   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 3.4e+00   | 3.4e+00   | 3.4e+00   | 3.4e+00   | 3.4e+00   | 5.0e-07   | 5.0e-07   | 5.0e-07   | 5.0e-07   | 5.0e-07   | 5.0e-01   | 5.0e-01   | 5.0e-01   | 5.0e-01   | 5.0e-01   
            2          | 4.5e-02   | 4.5e-02   | 4.5e-02   | 4.5e-02   | 4.5e-02   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 1.0e-06   | 5.0e-01   | 5.0e-01   | 5.0e-01   | 5.0e-01   | 5.0e-01   
            3          | 4.8e-06   | 4.8e-06   | 4.8e-06   | 4.8e-06   | 4.8e-06   | 5.0e-07   | 5.0e-07   | 5.0e-07   | 5.0e-07   | 5.0e-07   | 5.0e-01   | 5.0e-01   | 5.0e-01   | 5.0e-01   | 5.0e-01   
        >>> torch.allclose(theta_hat,theta_true,atol=5e-2)
        True
        >>> print_data_signatures(data)
            data['theta'].shape = (4,)
            data['iterrange'].shape = (4,)
            data['times'].shape = (4,)
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
            data['thetas'].shape = (4, 4)
            data['losses'].shape = (4,)
            data['lams'].shape = (4,)
            data['alphas'].shape = (4,)

    Batched example 
        
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
        ...     store_data_iters = None,
        ...     store_all_data = True,
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
            data['thetas'].shape = (3, 4, 4, 2, 2)
            data['losses'].shape = (3, 4, 4)
            data['lams'].shape = (3, 4, 4)
            data['alphas'].shape = (3, 4, 4)
    """
    if warn and (not torch.get_default_dtype()==torch.float64): warnings.warn('''
            torch.get_default_dtype() = %s, but lm_opt often requires high precision updates. We recommend using:
                torch.set_default_dtype(torch.float64)'''%str(torch.get_default_dtype()))
    assert torch.get_default_dtype() in [torch.float32,torch.float64]
    default_dtype = torch.get_default_dtype()
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
    assert verbose%1==0
    assert verbose>=0 
    if store_data_iters is None: 
        store_data_iters = max(1,iters//1000)
    assert store_data_iters%1==0
    assert store_data_iters>=0 
    assert isinstance(store_all_data,bool)
    assert isinstance(minimize,bool)
    signminimize = 1 if minimize else -1
    loss_mult = float(loss_mult)
    loss_shift = float(loss_shift)
    if residtol is None: 
        if default_dtype==torch.float64:
            residtol = 1e-12
        elif default_dtype==torch.float32:
            residtol = 2.5e-4
        else:
            raise Exception("default_dtype = %s not parsed"%str(default_dtype))
    assert residtol>=0
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
    assert verbose_indent%1==0 
    assert verbose_indent>=0
    assert isinstance(verbose_times,bool)
    if store_data_iters:
        iterrange = []
        times = []
        losses = []
        losses_quantiles = {str(qt):[] for qt in quantiles_losses}
        lams_quantiles = {str(qt):[] for qt in quantiles_lams}
        alphas_quantiles = {str(qt):[] for qt in quantiles_alphas}
        if store_all_data:
            thetas = []
            lams = []
            alphas = []
    def f_resid(theta, *f_kwargs_vec_vals):
        assert len(f_kwargs_vec_vals)==len(f_kwargs_vec_names)
        ytrue = f_kwargs_vec_vals[0]
        f_kwargs_vec = {f_kwargs_vec_names[i]:f_kwargs_vec_vals[i] for i in range(1,len(f_kwargs_vec_names))}
        all_args = f(theta,**f_kwargs_vec,**f_kwargs_no_vec)
        assert isinstance(all_args,tuple) or isinstance(all_args,torch.Tensor), "f must return a tuple or torch.Tensor"
        if isinstance(all_args,torch.Tensor):
            yhat = all_args
            args = ()
        else: #  isinstance(all_args,tuple)
            assert all(isinstance(arg,torch.Tensor) for arg in all_args)
            yhat = all_args[0]
            args = all_args[1:]
        resid = yhat-ytrue
        return resid,(resid,yhat,*args)
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
            _,(resid,yhat,*args) = f_resid(theta,*f_kwargs_vec_vals)
        else:
            (Jfull,),(resid,yhat,*args) = vjac_ftilde(theta,*f_kwargs_vec_vals)
        assert Jfull.shape==(R,*nonbatch_y_shape,*nonbatch_theta_shape)
        breakcond = i==iters or resid.abs().amax()<=residtol
        loss = loss_mult*(resid**2).flatten(start_dim=1).sum(-1)+loss_shift
        times_i = timer.toc()
        losses_quantiles_i = {str(qt): loss.nanquantile(qt/100) for qt in quantiles_losses}
        lams_quantiles_i = {str(qt): lam.nanquantile(qt/100) for qt in quantiles_lams}
        alphas_quantiles_i = {str(qt): alpha.nanquantile(qt/100) for qt in quantiles_alphas}
        if store_data_iters and (i%store_data_iters==0 or breakcond):
            iterrange.append(i)
            losses.append(loss.reshape(batch_shape).to(default_device))
            times.append(times_i)
            for qt in quantiles_losses:
                losses_quantiles[str(qt)].append(losses_quantiles_i[str(qt)].to(default_device))
            for qt in quantiles_lams:
                lams_quantiles[str(qt)].append(lams_quantiles_i[str(qt)].to(default_device))
            for qt in quantiles_alphas:
                alphas_quantiles[str(qt)].append(alphas_quantiles_i[str(qt)].to(default_device))
            if store_all_data:
                thetas.append(theta.reshape(*batch_shape,*nonbatch_theta_shape).to(default_device))
                lams.append(lam.reshape(batch_shape).to(default_device))
                alphas.append(alpha.reshape(batch_shape).to(default_device))
        if verbose and (i%verbose==0 or i==iters):
            _s_iter = "%-10d "%i
            _s_losses_qt = ("| %-9.1e "*len(verbose_quantiles_losses))%tuple(losses_quantiles_i[str(qt)] for qt in verbose_quantiles_losses)
            _s_lams_qt = ("| %-9.1e "*len(verbose_quantiles_lams))%tuple(lams_quantiles_i[str(qt)] for qt in verbose_quantiles_lams)
            _s_alphas_qt = ("| %-9.1e "*len(verbose_quantiles_alphas))%tuple(alphas_quantiles_i[str(qt)] for qt in verbose_quantiles_alphas)
            _s_times = "| %-10.1f "%(times_i) if verbose_times else ""
            print(" "*verbose_indent+_s_iter+_s_losses_qt+_s_lams_qt+_s_alphas_qt+_s_times)
        if breakcond: break
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
        thetasf_new[:,success] = thetasf[success]-signminimize*alpha_factors_i[:,None,None]*deltaf[success]
        thetas_new = thetasf_new.reshape((Q_alphas,Q_lams,R,*nonbatch_theta_shape))
        f_kwargs_vec_vals_success = [(torch.ones((Q_alphas,Q_lams)+(1,)*f_kwargs_vec_vals[l].ndim,device=device)*f_kwargs_vec_vals[l][None,None,...])[:,success] for l in range(len(f_kwargs_vec_vals))]
        residf_new = torch.inf*torch.ones((Q_alphas,Q_lams,R,K),device=device)
        _,(resid_new_success,*_) = f_resid(thetas_new[:,success],*f_kwargs_vec_vals_success)
        residf_new[:,success] = resid_new_success.reshape((Q_alphas,resid_new_success.size(1),K))
        losses_new = signminimize*(residf_new**2).sum(-1)
        ibest = losses_new.reshape((Q_alphas*Q_lams,R)).argmin(0) # (R,)
        ibest_alpha,ibest_lam = ibest//Q_lams,ibest%Q_lams
        lam_best_new = lams_try[ibest_lam,Rrange] # (R,)
        alpha_best_new = alphas_try[ibest_alpha,Rrange] # (R,)
        loss_best_new = losses_new[ibest_alpha,ibest_lam,Rrange] # (R,)
        thetas_best_new = thetas_new[ibest_alpha,ibest_lam,Rrange] # (R,*nonbatch_theta_shape)
        improved = loss_best_new<loss # (R,)
        lam[improved] = lam_best_new[improved]
        alpha[improved] = alpha_best_new[improved]
        theta[improved] = thetas_best_new[improved]
    theta = theta.reshape((*batch_shape,*nonbatch_theta_shape))
    if batch_shape==():
        args = [arg[0] for arg in args]
    if store_data_iters==0:
        if args==[]:
            return theta
        else:
            return theta,*args
    else:
        data = {
            "theta": theta.to(default_device), 
            "iterrange": torch.tensor(iterrange,dtype=int), 
            "times": torch.tensor(times), 
            "losses_quantiles": {str(qt):torch.tensor(losses_quantiles[str(qt)]) for qt in quantiles_losses},
            "lams_quantiles": {str(qt):torch.tensor(lams_quantiles[str(qt)]) for qt in quantiles_lams},
            "alphas_quantiles": {str(qt):torch.tensor(alphas_quantiles[str(qt)]) for qt in quantiles_alphas},
            }
        if store_all_data:
            data["thetas"] = torch.stack(thetas,dim=0)
            data["losses"] = torch.stack(losses,dim=0)
            data["lams"] = torch.stack(lams,dim=0)
            data["alphas"] = torch.stack(alphas,dim=0)
        if args==[]:
            return theta,data
        else:
            return theta,*args,data

def minres(
        A,
        B,
        X0 = None,
        iters = None,
        residtol = None,
        verbose = False, 
        verbose_indent = 4,
        quantiles_losses = [0,1,5,10,25,40,50,60,75,90,95,99,100],
        verbose_quantiles_losses = [5,25,50,75,90],
        verbose_times = True, 
        warn = True,
        store_data_iters = False, 
        store_all_data = False,
        ):
    r"""
    [MINRES algorithm](https://en.wikipedia.org/wiki/Minimal_residual_method) for solving linear systems $AX=B$ where $A$ is real-symmetric or complex-Hermitian

    A translation of [`scipy.sparse.linalg.minres`](https://github.com/scipy/scipy/blob/v1.17.0/scipy/sparse/linalg/_isolve/minres.py).

    Args:
        A (Union[torch.Tensor,callable]): Symmetric matrix `A` with shape `(...,n,n)`, or  
            `callable(A)` where `a(x)` should return the batch matrix multiplication of `A` and `X`,  
        B (torch.Tensor): Right hand side tensor $B$ with shape `(...,n,k)`
        X0 (torch.Tensor): Initial guess for $X$ with shape `(...,n,k)`, defaults to zeros. 
        iters (int): number of minres iterations, defaults to `5n`. 
        residtol (float): Non-negative tolerance on the maximum residual for early stopping, defaults to `1e-12` for `torch.float64` and `2.5e-4` for `torch.float32`.
        verbose (int): Controls logging verbosity

            - If `True`, perform logging. 
            - If a positive int, only log every `verbose` iterations. 
            - If `None`, set to a reasonable positive int based on the maximum number of iterations
            - If `False`, don't log. 
        
        verbose_indent (int): Non-negative number of indentation spaces for logging.
        quantiles_losses (list): Loss quantiles to record.
        verbose_quantiles_losses (list): Loss quantiles to show in verbose log.
        verbose_times (bool): If `False`, do not show the times in the verbose log. This is mostly for testing where timing is not reproducible. 
        warn (bool): If `False`, then suppress warnings.
        store_data_iters (int): Controls storage iterations with the same options as verbose. If `store_data_iters==0`, then the data is not collected or returned. 

            - If `True`, store every iteration. 
            - If a positive int, only store every `store_data_iters` iterations. 
            - If `None`, set to a reasonable positive int based on the maximum number of iterations
            - If `False`, don't store data, and do not return data 

        store_all_data (bool): If `True`, store the `x` values as well as the metrics. 

    Returns:
        x (torch.Tensor): Optimized $X$.
        data (dict): Iteration data, only returned when `store_data_iters>0`

    Examples:

        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

    Real-symmetric example with column vector $b$ 
        
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
    
    Complex-Hermitian example with column vector $b$ 
        
        >>> n = 5
        >>> A = torch.randn(n,n,dtype=torch.complex128,generator=rng)
        >>> A = (A+A.adjoint())/2
        >>> b = torch.rand(n,dtype=torch.complex128,generator=rng)
        >>> x_true = torch.linalg.solve(A,b[...,None])[...,0]
        >>> x_true
        tensor([ 0.2207+0.2879j,  0.0928-0.0057j,  0.2681+1.3488j, -1.2520-0.4214j,
                -0.8860-0.6922j])
        >>> torch.allclose(A@x_true-b,torch.zeros_like(b))
        True
        >>> x_minres = minres(A,b[...,None],verbose=None,verbose_times=False)[...,0]
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 8.7e-01   | 8.7e-01   | 8.7e-01   | 8.7e-01   | 8.7e-01   
            2          | 5.4e-01   | 5.4e-01   | 5.4e-01   | 5.4e-01   | 5.4e-01   
            3          | 5.4e-01   | 5.4e-01   | 5.4e-01   | 5.4e-01   | 5.4e-01   
            4          | 2.6e-01   | 2.6e-01   | 2.6e-01   | 2.6e-01   | 2.6e-01   
            5          | 3.0e-15   | 3.0e-15   | 3.0e-15   | 3.0e-15   | 3.0e-15   
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
        tensor([[ 0.8801, -0.0116,  0.4805],
                [-1.1095, -1.6166, -0.7103],
                [-2.9918, -1.9201, -3.5855],
                [-4.1777, -3.6586, -5.1658],
                [ 1.5417,  0.9814,  1.3790]])
        >>> torch.allclose(A@X_true-B,torch.zeros_like(B))
        True
        >>> X_minres = minres(A,B,verbose=None,verbose_times=False)
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 6.6e-01   | 7.5e-01   | 8.7e-01   | 8.8e-01   | 8.9e-01   
            2          | 6.2e-01   | 6.4e-01   | 6.6e-01   | 7.3e-01   | 7.6e-01   
            3          | 3.5e-01   | 4.7e-01   | 6.1e-01   | 6.3e-01   | 6.4e-01   
            4          | 1.5e-01   | 1.8e-01   | 2.2e-01   | 3.4e-01   | 4.1e-01   
            5          | 7.0e-15   | 9.1e-15   | 1.2e-14   | 2.2e-14   | 2.8e-14   
        >>> torch.allclose(X_minres,X_true)
        True

    Tri-diagonal $A$ with storage-saving multiplication function 
        
        >>> n = 5
        >>> k = 3
        >>> A_diag = torch.randn(n,generator=rng)
        >>> A_off_diag = torch.randn(n-1,generator=rng) 
        >>> A = torch.zeros(n,n)
        >>> A[torch.arange(n),torch.arange(n)] = A_diag 
        >>> A[torch.arange(n-1),torch.arange(1,n)] = A_off_diag
        >>> A[torch.arange(1,n),torch.arange(n-1)] = A_off_diag
        >>> A
        tensor([[-0.2728, -0.1545,  0.0000,  0.0000,  0.0000],
                [-0.1545, -0.0275, -0.0120,  0.0000,  0.0000],
                [ 0.0000, -0.0120, -0.4436,  0.2802,  0.0000],
                [ 0.0000,  0.0000,  0.2802, -0.7303,  0.9724],
                [ 0.0000,  0.0000,  0.0000,  0.9724, -0.4180]])
        >>> B = torch.rand(n,k,generator=rng)
        >>> X_true = torch.linalg.solve(A,B)
        >>> X_true
        tensor([[-5.5996, -3.1344, -6.0989],
                [ 6.5450, -0.5211,  5.0505],
                [-1.8303, -0.9818, -0.4647],
                [ 0.5862,  0.9233,  1.2342],
                [ 1.3614,  1.4523,  1.9166]])
        >>> torch.allclose(A@X_true-B,torch.zeros_like(B))
        True
        >>> def A_mult(x):
        ...     y = x*A_diag[:,None]
        ...     y[1:,:] += x[:-1,:]*A_off_diag[:,None]
        ...     y[:-1,:] += x[1:,:]*A_off_diag[:,None]
        ...     return y
        >>> torch.allclose(A_mult(X_true),A@X_true)
        True
        >>> X_minres = minres(A_mult,B,verbose=None,verbose_times=False)
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 7.5e-01   | 8.1e-01   | 8.8e-01   | 9.3e-01   | 9.6e-01   
            2          | 4.5e-01   | 5.6e-01   | 7.1e-01   | 8.0e-01   | 8.5e-01   
            3          | 1.5e-01   | 1.9e-01   | 2.4e-01   | 2.9e-01   | 3.2e-01   
            4          | 5.7e-02   | 1.4e-01   | 2.4e-01   | 2.9e-01   | 3.2e-01   
            5          | 4.0e-15   | 5.4e-15   | 7.1e-15   | 1.9e-14   | 2.6e-14   
        >>> torch.allclose(X_minres,X_true)
        True

    Batched tri-diagonal $A$ with storage-saving multiplication function 

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
        >>> X_minres,data = minres(A_mult,B,verbose=None,verbose_times=False,store_data_iters=None,store_all_data=True)
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            25         | 1.8e-01   | 2.3e-01   | 2.6e-01   | 2.9e-01   | 3.2e-01   
            50         | 9.4e-02   | 1.2e-01   | 1.5e-01   | 1.9e-01   | 2.1e-01   
            75         | 3.3e-02   | 5.1e-02   | 8.2e-02   | 1.5e-01   | 1.8e-01   
            100        | 7.8e-03   | 1.4e-02   | 3.8e-02   | 1.2e-01   | 1.5e-01   
            125        | 9.5e-07   | 1.2e-04   | 5.6e-04   | 2.3e-02   | 5.1e-02   
            150        | 1.2e-14   | 2.6e-14   | 1.7e-12   | 2.5e-10   | 3.1e-09   
            175        | 2.2e-15   | 3.7e-15   | 1.4e-14   | 6.5e-14   | 1.6e-13   
            176        | 2.2e-15   | 3.7e-15   | 1.3e-14   | 6.0e-14   | 1.4e-13   
        >>> X_minres.shape
        torch.Size([2, 6, 4, 100, 3])
        >>> torch.allclose(X_minres,X_true)
        True
        >>> print_data_signatures(data)
            data['x'].shape = (2, 6, 4, 100, 3)
            data['iterrange'].shape = (177,)
            data['times'].shape = (177,)
            data['losses_quantiles']
                data['losses_quantiles']['0'].shape = (177,)
                data['losses_quantiles']['1'].shape = (177,)
                data['losses_quantiles']['5'].shape = (177,)
                data['losses_quantiles']['10'].shape = (177,)
                data['losses_quantiles']['25'].shape = (177,)
                data['losses_quantiles']['40'].shape = (177,)
                data['losses_quantiles']['50'].shape = (177,)
                data['losses_quantiles']['60'].shape = (177,)
                data['losses_quantiles']['75'].shape = (177,)
                data['losses_quantiles']['90'].shape = (177,)
                data['losses_quantiles']['95'].shape = (177,)
                data['losses_quantiles']['99'].shape = (177,)
                data['losses_quantiles']['100'].shape = (177,)
            data['xs'].shape = (177, 2, 6, 4, 100, 3)
            data['losses'].shape = (177, 2, 6, 4, 3)
    """
    if warn and (not torch.get_default_dtype()==torch.float64): warnings.warn('''
            torch.get_default_dtype() = %s, but lm_opt often requires high precision updates. We recommend using:
                torch.set_default_dtype(torch.float64)'''%str(torch.get_default_dtype()))
    assert torch.get_default_dtype() in [torch.float32,torch.float64]
    default_dtype = torch.get_default_dtype()
    device = str(B.device)
    default_device = str(torch.get_default_device())
    assert B.ndim>=2, "B should have shape (...,n,k)"
    n = B.size(-2)
    k = B.size(-1)
    if X0 is None: 
        X0 = torch.zeros_like(B)
    if isinstance(A,torch.Tensor):
        assert A.shape[-2:]==(n,n)
        assert torch.allclose(A.adjoint(),A)
        matvec = lambda X: torch.einsum("...ij,...jk->...ik",A,X)
    else:
        assert callable(A)
        matvec = A
    if iters is None: 
        iters = 5*n 
    assert iters>=0
    assert iters%1==0
    if residtol is None: 
        if default_dtype==torch.float64:
            residtol = 1e-12
        elif default_dtype==torch.float32:
            residtol = 2.5e-4
        else:
            raise Exception("default_dtype = %s not parsed"%str(default_dtype))
    assert residtol>=0
    if verbose is None: 
        verbose = max(1,iters//20)
    assert verbose%1==0
    assert verbose>=0 
    if store_data_iters is None: 
        store_data_iters = max(1,iters//1000)
    assert store_data_iters%1==0
    assert store_data_iters>=0 
    assert isinstance(store_all_data,bool)
    assert isinstance(quantiles_losses,list)
    assert all(0<=qt<=100 for qt in quantiles_losses)
    assert isinstance(verbose_quantiles_losses,list)
    assert all(qt in quantiles_losses for qt in verbose_quantiles_losses)
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
    inner = lambda a,b: torch.einsum("...ij,...ij->...j",a.conj(),b)
    Anorm = 0
    eps = torch.finfo(B.dtype).eps
    x = X0 
    Ax = matvec(x)
    assert Ax.shape[-2:]==(n,k)
    batch_shape = tuple(Ax.shape[:-2])
    if store_data_iters:
        iterrange = []
        times = []
        losses = []
        losses_quantiles = {str(qt):[] for qt in quantiles_losses}
        if store_all_data:
            xs = []
    r1 = B-Ax # (...,n,k)
    y = psolve(r1) # (...,n,k)
    beta1 = torch.sqrt(inner(r1,y)) # (...,k)
    bnorm = torch.linalg.norm(B,dim=-2) # (...,k)
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = torch.zeros(1,device=device)
    phibar = beta1
    tnorm2 = 0
    cs = -1
    sn = 0
    w = torch.zeros_like(B)
    w2 = torch.zeros_like(B)
    r2 = r1
    shift = 0 # TODO: If shift != 0 then the method solves (A - shift*I)x = b
    for i in range(iters+1):
        resid = matvec(x)-B 
        breakcond = i==iters or resid.abs().amax()<=residtol
        loss = torch.linalg.norm(resid,dim=-2)/bnorm
        times_i = timer.toc()
        losses_quantiles_i = {str(qt): loss.nanquantile(qt/100) for qt in quantiles_losses}
        if store_data_iters and (i%store_data_iters==0 or breakcond):
            iterrange.append(i)
            losses.append(loss.to(default_device))
            times.append(times_i)
            for qt in quantiles_losses:
                losses_quantiles[str(qt)].append(losses_quantiles_i[str(qt)].to(default_device))
            if store_all_data:
                xs.append(x.expand(resid.shape).to(default_device))
        if verbose and (i%verbose==0 or breakcond):
            _s_iter = "%-10d "%i
            _s_losses_qt = ("| %-9.1e "*len(verbose_quantiles_losses))%tuple(losses_quantiles_i[str(qt)] for qt in verbose_quantiles_losses)
            _s_times = "| %-10.1f "%(times_i) if verbose_times else ""
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
        beta = torch.sqrt(beta)
        tnorm2 += alfa**2+oldb**2+beta**2
        oldeps = epsln
        delta = cs*dbar+sn*alfa
        gbar = sn*dbar-cs*alfa
        epsln = sn*beta
        dbar = -cs*beta
        gamma = torch.linalg.norm(torch.stack([gbar,beta],dim=-1),dim=-1)
        gamma = torch.maximum(gamma,eps*torch.ones(1,device=device))
        cs = gbar/gamma
        sn = beta/gamma
        phi = cs*phibar
        phibar = sn*phibar
        denom = 1/gamma
        w1 = w2
        w2 = w
        w = (v-oldeps[...,None,:]*w1-delta[...,None,:]*w2)*denom[...,None,:]
        x = x+phi[...,None,:]*w
    if store_data_iters==0:
        return x 
    else:
        data = {
            "x": x.to(default_device), 
            "iterrange": torch.tensor(iterrange,dtype=int), 
            "times": torch.tensor(times), 
            "losses_quantiles": {str(qt):torch.tensor(losses_quantiles[str(qt)]) for qt in quantiles_losses},
            }
        if store_all_data:
            data["xs"] = torch.stack(xs,dim=0)
            data["losses"] = torch.stack(losses,dim=0)
        return x,data

def minres_qlp_cs(
        A,
        B,
        X0 = None,
        iters = None,
        residtol = None,
        verbose = False, 
        verbose_indent = 4,
        quantiles_losses = [0,1,5,10,25,40,50,60,75,90,95,99,100],
        verbose_quantiles_losses = [5,25,50,75,90],
        verbose_times = True, 
        warn = True,
        store_data_iters = False, 
        store_all_data = False,
        ):
    # https://github.com/schoi32/sci498rms/blob/master/Sorokin_MinresQLP_Python_Workspace/MinresQLP/Algorithms/cs_mqlp.py
    r"""
    MINRES QLP algorith for complex-symmetric matrices. 

    A translation of the [MATLAB version of MINRESQLP](http://www.stanford.edu/group/SOL/software.html).

    References 

    1.  S.-C. Choi, C. C. Paige, and M. A. Saunders,
        MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric systems,
        SIAM Journal of Scientific Computing, submitted on March 7, 2010.

    2.  S.-C. Choi's PhD Dissertation, Stanford University, 2006: 
        http://www.stanford.edu/group/SOL/dissertations.html

    Args:
        A (Union[torch.Tensor,callable]): Symmetric matrix `A` with shape `(...,n,n)`, or  
            `callable(A)` where `a(x)` should return the batch matrix multiplication of `A` and `X`,  
        B (torch.Tensor): Right hand side tensor $B$ with shape `(...,n,k)`
        X0 (torch.Tensor): Initial guess for $X$ with shape `(...,n,k)`, defaults to zeros. 
        iters (int): number of minres iterations, defaults to `5n`. 
        residtol (float): Non-negative tolerance on the maximum residual for early stopping, defaults to `1e-12` for `torch.float64` and `2.5e-4` for `torch.float32`.
        verbose (int): Controls logging verbosity

            - If `True`, perform logging. 
            - If a positive int, only log every `verbose` iterations. 
            - If `None`, set to a reasonable positive int based on the maximum number of iterations
            - If `False`, don't log. 
        
        verbose_indent (int): Non-negative number of indentation spaces for logging.
        quantiles_losses (list): Loss quantiles to record.
        verbose_quantiles_losses (list): Loss quantiles to show in verbose log.
        verbose_times (bool): If `False`, do not show the times in the verbose log. This is mostly for testing where timing is not reproducible. 
        warn (bool): If `False`, then suppress warnings.
        store_data_iters (int): Controls storage iterations with the same options as verbose. If `store_data_iters==0`, then the data is not collected or returned. 

            - If `True`, store every iteration. 
            - If a positive int, only store every `store_data_iters` iterations. 
            - If `None`, set to a reasonable positive int based on the maximum number of iterations
            - If `False`, don't store data, and do not return data 

        store_all_data (bool): If `True`, store the `x` values as well as the metrics. 

    Returns:
        x (torch.Tensor): Optimized $X$.
        data (dict): Iteration data, only returned when `store_data_iters>0`

    Examples:

        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

    Column vector $b$ 
        
        >>> n = 5
        >>> A = torch.randn(n,n,dtype=torch.complex128,generator=rng)
        >>> A = (A+A.T)/2
        >>> b = torch.rand(n,dtype=torch.complex128,generator=rng)
        >>> x_true = torch.linalg.solve(A,b[...,None])[...,0]
        >>> x_true
        tensor([-0.6207-0.4121j,  0.5221+0.3249j, -1.0952+0.8594j,  0.9080-1.2110j,
                -0.9799+0.7372j])
        >>> torch.allclose(A@x_true-b,torch.zeros_like(b))
        True
        >>> x_minres = minres_qlp_cs(A,b[...,None],verbose=None,verbose_times=False)[...,0]
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 9.6e-01   | 9.6e-01   | 9.6e-01   | 9.6e-01   | 9.6e-01   
            2          | 5.8e-01   | 5.8e-01   | 5.8e-01   | 5.8e-01   | 5.8e-01   
            3          | 4.5e-01   | 4.5e-01   | 4.5e-01   | 4.5e-01   | 4.5e-01   
            4          | 2.8e-01   | 2.8e-01   | 2.8e-01   | 2.8e-01   | 2.8e-01   
            5          | 1.7e-14   | 1.7e-14   | 1.7e-14   | 1.7e-14   | 1.7e-14   
        >>> torch.allclose(x_minres,x_true)
        True

    Matrix $B$
        
        >>> n = 5
        >>> k = 3
        >>> A = torch.randn(n,n,dtype=torch.complex128,generator=rng)
        >>> A = (A+A.T)/2
        >>> B = torch.rand(n,k,dtype=torch.complex128,generator=rng)
        >>> X_true = torch.linalg.solve(A,B)
        >>> X_true
        tensor([[ 0.0142+0.7190j,  0.0097+0.9734j, -0.3620+0.6413j],
                [ 0.4527+0.7455j,  0.4270+0.4941j,  0.9255+0.5685j],
                [-0.7182-1.0284j,  0.3193-0.4463j, -0.3421-0.8780j],
                [-0.5973+0.4910j, -0.8147+0.8256j, -0.8821+0.7147j],
                [-0.0926+0.5500j, -0.5192+0.2817j, -0.8496+0.6976j]])
        >>> torch.allclose(A@X_true-B,torch.zeros_like(B))
        True
        >>> X_minres = minres_qlp_cs(A,B,verbose=None,verbose_times=False)
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 9.0e-01   | 9.1e-01   | 9.2e-01   | 9.5e-01   | 9.7e-01   
            2          | 2.2e-01   | 3.1e-01   | 4.3e-01   | 5.0e-01   | 5.5e-01   
            3          | 2.0e-01   | 2.2e-01   | 2.3e-01   | 2.9e-01   | 3.3e-01   
            4          | 3.4e-02   | 5.0e-02   | 7.1e-02   | 8.1e-02   | 8.7e-02   
            5          | 1.9e-15   | 4.2e-15   | 7.0e-15   | 1.5e-14   | 2.0e-14   
        >>> torch.allclose(X_minres,X_true)
        True

    Tri-diagonal $A$ with storage-saving multiplication function 
        
        >>> n = 4
        >>> k = 3
        >>> A_diag = torch.randn(n,dtype=torch.complex128,generator=rng)
        >>> A_off_diag = torch.randn(n-1,dtype=torch.complex128,generator=rng) 
        >>> A = torch.zeros(n,n,dtype=torch.complex128)
        >>> A[torch.arange(n),torch.arange(n)] = A_diag 
        >>> A[torch.arange(n-1),torch.arange(1,n)] = A_off_diag
        >>> A[torch.arange(1,n),torch.arange(n-1)] = A_off_diag
        >>> A
        tensor([[ 0.4070+0.4993j, -0.3137-0.5164j,  0.0000+0.0000j,  0.0000+0.0000j],
                [-0.3137-0.5164j, -0.2736+0.1860j, -0.2956-0.1092j,  0.0000+0.0000j],
                [ 0.0000+0.0000j, -0.2956-0.1092j,  0.4033-0.3862j, -0.0085+0.1981j],
                [ 0.0000+0.0000j,  0.0000+0.0000j, -0.0085+0.1981j, -0.1929-0.0194j]])
        >>> B = torch.rand(n,k,dtype=torch.complex128,generator=rng)
        >>> X_true = torch.linalg.solve(A,B)
        >>> X_true
        tensor([[-0.0284-1.1816j, -0.6400-0.1975j,  0.7656-1.1306j],
                [-1.6586-0.3570j, -2.4720-0.1880j, -1.1169-0.6640j],
                [-2.4585-1.2378j, -1.0485-2.0358j, -3.1323-0.3632j],
                [ 0.1591-7.4823j,  1.8120-2.6001j, -2.3910-7.6943j]])
        >>> torch.allclose(A@X_true-B,torch.zeros_like(B))
        True
        >>> def A_mult(x):
        ...     y = x*A_diag[:,None]
        ...     y[1:,:] += x[:-1,:]*A_off_diag[:,None]
        ...     y[:-1,:] += x[1:,:]*A_off_diag[:,None]
        ...     return y
        >>> torch.allclose(A_mult(X_true),A@X_true)
        True
        >>> X_minres = minres_qlp_cs(A_mult,B,verbose=None,verbose_times=False)
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            1          | 7.8e-01   | 8.3e-01   | 8.9e-01   | 9.4e-01   | 9.7e-01   
            2          | 5.1e-01   | 5.2e-01   | 5.2e-01   | 5.4e-01   | 5.5e-01   
            3          | 3.2e-01   | 3.2e-01   | 3.2e-01   | 3.2e-01   | 3.3e-01   
            4          | 9.1e-16   | 1.1e-15   | 1.3e-15   | 2.1e-15   | 2.6e-15   
        >>> torch.allclose(X_minres,X_true)
        True

    Batched tri-diagonal $A$ with storage-saving multiplication function 

        >>> n = 100
        >>> k = 3
        >>> A_diag = torch.randn(2,1,4,n,dtype=torch.complex128,generator=rng)
        >>> A_off_diag = torch.randn(2,1,4,n-1,dtype=torch.complex128,generator=rng) 
        >>> A = torch.zeros(2,1,4,n,n,dtype=torch.complex128)
        >>> A[...,torch.arange(n),torch.arange(n)] = A_diag 
        >>> A[...,torch.arange(n-1),torch.arange(1,n)] = A_off_diag
        >>> A[...,torch.arange(1,n),torch.arange(n-1)] = A_off_diag
        >>> B = torch.rand(2,6,1,n,k,dtype=torch.complex128,generator=rng)
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
        >>> X_minres,data = minres_qlp_cs(A_mult,B,verbose=None,verbose_times=False,store_data_iters=None,store_all_data=True,iters=40)
            iter i     | losses_quantiles                                          
                       | 5         | 25        | 50        | 75        | 90        
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            0          | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   | 1.0e+00   
            2          | 6.6e-01   | 6.8e-01   | 7.0e-01   | 7.2e-01   | 7.3e-01   
            4          | 5.1e-01   | 5.4e-01   | 5.6e-01   | 5.8e-01   | 5.9e-01   
            6          | 4.2e-01   | 4.5e-01   | 4.6e-01   | 4.9e-01   | 5.0e-01   
            8          | 3.5e-01   | 3.8e-01   | 4.0e-01   | 4.2e-01   | 4.4e-01   
            10         | 3.0e-01   | 3.3e-01   | 3.6e-01   | 3.8e-01   | 4.0e-01   
            12         | 2.6e-01   | 2.9e-01   | 3.2e-01   | 3.4e-01   | 3.6e-01   
            14         | 2.3e-01   | 2.6e-01   | 2.9e-01   | 3.1e-01   | 3.3e-01   
            16         | 2.1e-01   | 2.3e-01   | 2.6e-01   | 2.8e-01   | 3.0e-01   
            18         | 1.9e-01   | 2.1e-01   | 2.4e-01   | 2.6e-01   | 2.8e-01   
            20         | 1.7e-01   | 1.9e-01   | 2.2e-01   | 2.3e-01   | 2.6e-01   
            22         | 1.5e-01   | 1.7e-01   | 2.0e-01   | 2.2e-01   | 2.4e-01   
            24         | 1.3e-01   | 1.6e-01   | 1.8e-01   | 2.0e-01   | 2.2e-01   
            26         | 1.2e-01   | 1.4e-01   | 1.7e-01   | 1.9e-01   | 2.1e-01   
            28         | 1.0e-01   | 1.3e-01   | 1.6e-01   | 1.8e-01   | 2.0e-01   
            30         | 9.3e-02   | 1.2e-01   | 1.4e-01   | 1.7e-01   | 1.9e-01   
            32         | 8.4e-02   | 1.1e-01   | 1.3e-01   | 1.6e-01   | 1.9e-01   
            34         | 7.6e-02   | 1.0e-01   | 1.2e-01   | 1.6e-01   | 1.8e-01   
            36         | 6.9e-02   | 9.3e-02   | 1.1e-01   | 1.5e-01   | 1.7e-01   
            38         | 6.4e-02   | 8.5e-02   | 1.0e-01   | 1.4e-01   | 1.7e-01   
            40         | 5.8e-02   | 7.5e-02   | 9.2e-02   | 1.4e-01   | 1.6e-01   
        >>> X_minres.shape
        torch.Size([2, 6, 4, 100, 3])
        >>> torch.allclose(X_minres,X_true)
        False
        >>> print_data_signatures(data)
            data['x'].shape = (2, 6, 4, 100, 3)
            data['iterrange'].shape = (41,)
            data['times'].shape = (41,)
            data['losses_quantiles']
                data['losses_quantiles']['0'].shape = (41,)
                data['losses_quantiles']['1'].shape = (41,)
                data['losses_quantiles']['5'].shape = (41,)
                data['losses_quantiles']['10'].shape = (41,)
                data['losses_quantiles']['25'].shape = (41,)
                data['losses_quantiles']['40'].shape = (41,)
                data['losses_quantiles']['50'].shape = (41,)
                data['losses_quantiles']['60'].shape = (41,)
                data['losses_quantiles']['75'].shape = (41,)
                data['losses_quantiles']['90'].shape = (41,)
                data['losses_quantiles']['95'].shape = (41,)
                data['losses_quantiles']['99'].shape = (41,)
                data['losses_quantiles']['100'].shape = (41,)
            data['xs'].shape = (41, 2, 6, 4, 100, 3)
            data['losses'].shape = (41, 2, 6, 4, 3)
    """
    if warn and (not torch.get_default_dtype()==torch.float64): warnings.warn('''
            torch.get_default_dtype() = %s, but lm_opt often requires high precision updates. We recommend using:
                torch.set_default_dtype(torch.float64)'''%str(torch.get_default_dtype()))
    assert torch.get_default_dtype() in [torch.float32,torch.float64]
    default_dtype = torch.get_default_dtype()
    device = str(B.device)
    default_device = str(torch.get_default_device())
    assert B.ndim>=2, "B should have shape (...,n,k)"
    n = B.size(-2)
    k = B.size(-1)
    assert B.dtype in [torch.complex64,torch.complex128]
    if X0 is None: 
        X0 = torch.zeros_like(B)
    if isinstance(A,torch.Tensor):
        assert A.shape[-2:]==(n,n)
        assert torch.allclose(A.T,A)
        matvec = lambda X: torch.einsum("...ij,...jk->...ik",A,X)
    else:
        assert callable(A)
        matvec = A
    if iters is None: 
        iters = 5*n 
    assert iters>=0
    assert iters%1==0
    if residtol is None: 
        if default_dtype==torch.float64:
            residtol = 1e-12
        elif default_dtype==torch.float32:
            residtol = 2.5e-4
        else:
            raise Exception("default_dtype = %s not parsed"%str(default_dtype))
    assert residtol>=0
    if verbose is None: 
        verbose = max(1,iters//20)
    assert verbose%1==0
    assert verbose>=0 
    if store_data_iters is None: 
        store_data_iters = max(1,iters//1000)
    assert store_data_iters%1==0
    assert store_data_iters>=0 
    assert isinstance(store_all_data,bool)
    assert isinstance(quantiles_losses,list)
    assert all(0<=qt<=100 for qt in quantiles_losses)
    assert isinstance(verbose_quantiles_losses,list)
    assert all(qt in quantiles_losses for qt in verbose_quantiles_losses)
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
    inner = lambda a,b: torch.einsum("...ij,...ij->...j",a.conj(),b)
    Anorm = 0
    x = X0 
    Ax = matvec(x)
    assert Ax.shape[-2:]==(n,k)
    batch_shape = tuple(Ax.shape[:-2])
    if store_data_iters:
        iterrange = []
        times = []
        losses = []
        losses_quantiles = {str(qt):[] for qt in quantiles_losses}
        if store_all_data:
            xs = []
    r2 = B  # (...,k)
    r3 = r2 # (...,k)
    r3 = psolve(r2)
    beta1 = torch.sqrt(inner(r3,r2))  # (...,k)
    bnorm = torch.linalg.norm(B,dim=-2) # (...,k)
    # TODO: Check if below variables are necessary    
    beta = torch.zeros_like(beta1)
    tau = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    taul = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    phi = beta1
    betan = beta1
    cs = -torch.ones((*batch_shape,k),dtype=B.dtype,device=device)
    sn = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    cr1 = torch.ones((*batch_shape,k),dtype=B.dtype,device=device)
    sr1 = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    cr2 = -torch.ones((*batch_shape,k),dtype=B.dtype,device=device)
    sr2 = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    dltan = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    eplnn = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    gama = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    gamal = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    gamal2 = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    eta = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    etal = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    etal2 = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    vepln = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    veplnl = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    veplnl2 = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    ul3 = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    ul2 = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    ul = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    u = torch.zeros((*batch_shape,k),dtype=B.dtype,device=device)
    w = torch.zeros_like(B)
    wl = torch.zeros_like(B)
    r1 = torch.zeros_like(B)
    xl2 = torch.zeros_like(B)
    alfa = torch.zeros_like(B)
    shift = 0 # TODO: If shift != 0 then the method solves (A - shift*I)x = b
    for i in range(iters+1):
        resid = matvec(x)-B 
        breakcond = i==iters or resid.abs().amax()<=residtol
        loss = torch.linalg.norm(resid,dim=-2)/bnorm
        times_i = timer.toc()
        losses_quantiles_i = {str(qt): loss.nanquantile(qt/100) for qt in quantiles_losses}
        if store_data_iters and (i%store_data_iters==0 or breakcond):
            iterrange.append(i)
            losses.append(loss.to(default_device))
            times.append(times_i)
            for qt in quantiles_losses:
                losses_quantiles[str(qt)].append(losses_quantiles_i[str(qt)].to(default_device))
            if store_all_data:
                xs.append(x.expand(resid.shape).to(default_device))
        if verbose and (i%verbose==0 or breakcond):
            _s_iter = "%-10d "%i
            _s_losses_qt = ("| %-9.1e "*len(verbose_quantiles_losses))%tuple(losses_quantiles_i[str(qt)] for qt in verbose_quantiles_losses)
            _s_times = "| %-10.1f "%(times_i) if verbose_times else ""
            print(" "*verbose_indent+_s_iter+_s_losses_qt+_s_times)
        if breakcond: break 
        betal = beta
        beta = betan
        v = r3/beta[...,None,:]    
        r3 = matvec(v.conj())-shift*v.conj()
        if i>0:
            r3 = r3-(beta/betal)[...,None,:]*r1
        alfa = inner(v,r3)
        r3 = r3-(alfa/beta)[...,None,:]*r2
        r1 = r2
        r2 = r3
        r3 = psolve(r2)
        betan = torch.sqrt(inner(r2,r3))
        dbar = dltan
        dlta = cs*dbar+sn*alfa
        epln = eplnn
        gbar = sn.conj()*dbar-cs*alfa
        eplnn = sn*betan
        dltan = -cs*betan
        dlta_QLP = dlta
        gamal3 = gamal2
        gamal2 = gamal
        gamal = gama
        (cs,sn,gama) = symOrtho(gbar,betan)
        gama_tmp = gama
        taul2  = taul
        taul = tau
        tau = cs*phi
        phi = sn.conj()*phi
        if i>1:
            veplnl2  = veplnl
            etal2 = etal
            etal = eta
            dlta_tmp = sr2*vepln - cr2*dlta
            veplnl = cr2*vepln+sr2.conj()*dlta
            dlta = dlta_tmp
            eta = sr2.conj()*gama
            gama = -cr2*gama
        if i>0:
            (cr1,sr1,gamal) = symOrtho(gamal.conj(),dlta.conj())
            gamal = gamal.conj()
            vepln = sr1.conj()*gama
            gama = -cr1*gama
        ul4 = ul3
        ul3 = ul2
        if i>1:
            ul2 = (taul2-etal2*ul4-veplnl2*ul3)/gamal2
        if i>0:
            ul = (taul-etal*ul3-veplnl*ul2)/gamal
        u = (tau - eta*ul2 - vepln*ul) / gama
        if i==0:
            wl2 = wl
            wl = v.conj()*sr1.conj()[...,None,:]
            w  = v.conj()*cr1[...,None,:]
        elif i==1:
            wl2 = wl
            wl = w*cr1[...,None,:]+v.conj()*sr1.conj()[...,None,:]
            w = w*sr1[...,None,:]-v.conj()*cr1[...,None,:]
        else:
            wl2 = wl
            wl = w
            w  = wl2*sr2[...,None,:]-v.conj()*cr2[...,None,:]
            wl2 = wl2*cr2[...,None,:]+v.conj()*sr2.conj()[...,None,:]
            v = wl*cr1[...,None,:]+w*sr1.conj()[...,None,:]
            w = wl*sr1[...,None,:]-w*cr1[...,None,:]
            wl = v
        xl2 = xl2+wl2*ul2[...,None,:]
        x = xl2+wl*ul[...,None,:]+w*u[...,None,:]
        pass
        (cr2,sr2,gamal) = symOrtho(gamal.conj(),eplnn.conj())
        gamal = gamal.conj()
    if store_data_iters==0:
        return x 
    else:
        data = {
            "x": x.to(default_device), 
            "iterrange": torch.tensor(iterrange,dtype=int), 
            "times": torch.tensor(times), 
            "losses_quantiles": {str(qt):torch.tensor(losses_quantiles[str(qt)]) for qt in quantiles_losses},
            }
        if store_all_data:
            data["xs"] = torch.stack(xs,dim=0)
            data["losses"] = torch.stack(losses,dim=0)
        return x,data

def sign(x):
    a = x.real
    b = x.imag
    c = torch.sqrt(a**2+b**2)
    s = torch.where(c==0,torch.zeros_like(a),(a/c)+(b/c)*1j)
    return s

def symOrtho(a, b):    
    absa = a.abs()
    absb = b.abs()
    signa = sign(a)
    signb = sign(b)
    c = torch.zeros_like(a)
    s = torch.zeros_like(a)
    r = torch.zeros_like(a)
    b0 = b==0
    c[b0] = 1 
    s[b0] = 0 
    r[b0] = a[b0]
    a0 = (b!=0)*(a==0)
    c[a0] = 0 
    s[a0] = 1 
    r[a0] = b[a0]
    cond1 = (absb>absa)*(b!=0)*(a!=0)
    t1 = absa[cond1]/absb[cond1]
    c[cond1] = 1/torch.sqrt(1+t1**2)+0j
    s[cond1] = c[cond1]*(signb[cond1]/signa[cond1]).conj()
    c[cond1] = c[cond1]*t1
    r[cond1] = b[cond1]/s[cond1].conj()
    cond2 = (absb<=absa)*(b!=0)*(a!=0)
    t2 = absb[cond2]/absa[cond2]
    c[cond2] = 1/torch.sqrt(1+t2**2)+0j
    s[cond2] = c[cond2]*t2*(signb[cond2]/signa[cond2]).conj()
    r[cond2] = a[cond2]/c[cond2]
    return (c,s,r)
