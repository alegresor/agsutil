import torch 

def gradb(f, x, bkwargs={}, bdims=0, chunk_size=None):
    r"""
    Batched `torch.func.grad` and function evaluation 

    Args:
        f (callable): Function to compute `torch.func.grad` of. 
        x (Union[torch.Tensor,Tuple]): (batched) `Torch.Tensor` items whose gradients will be computed.
        bkwargs (dict): (batched) `Torch.Tensor` items whose gradients will not be computed.
        bdims (int): number of batch dimensions 
        chunk_size (int): to be passed into `torch.func.vmap`. 
    
    Returns: 
        gradys (tuple): `torch.Tensor` gradients, one with respect to each item in `x`. 
        y (torch.Tensor): Function evaluations. 
    
    Examples: 
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

        >>> f = lambda x: (x**2).sum(-1)
        >>> x = torch.rand(5,generator=rng) 
        >>> (grady,),y = gradb(f,x)
        >>> grady.shape 
        torch.Size([5])
        >>> y.shape 
        torch.Size([])
        >>> torch.allclose(y,f(x))
        True
        >>> torch.allclose(grady,2*x)
        True

        >>> f = lambda x,z: (x**2*z**2).sum(-1)
        >>> x = torch.rand(5,generator=rng) 
        >>> z = torch.rand(5,generator=rng) 
        >>> (grady_x,grady_z),y = gradb(f,(x,z))
        >>> grady_x.shape 
        torch.Size([5])
        >>> grady_z.shape 
        torch.Size([5])
        >>> y.shape
        torch.Size([])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(grady_x,2*x*z**2)
        True
        >>> torch.allclose(grady_z,x**2*2*z)
        True

        >>> f = lambda x: (x**2).sum(-1)
        >>> x = torch.rand(3,4,5,generator=rng) 
        >>> (grady,),y = gradb(f,x,bdims=2)
        >>> grady.shape 
        torch.Size([3, 4, 5])
        >>> y.shape 
        torch.Size([3, 4])
        >>> torch.allclose(y,f(x))
        True
        >>> torch.allclose(grady,2*x)
        True

        >>> f = lambda x,z: (x**2*z**2).sum(-1)
        >>> x = torch.rand(3,4,5,generator=rng) 
        >>> z = torch.rand(3,4,5,generator=rng) 
        >>> (grady_x,grady_z),y = gradb(f,(x,z),bdims=2)
        >>> grady_x.shape 
        torch.Size([3, 4, 5])
        >>> grady_z.shape 
        torch.Size([3, 4, 5])
        >>> y.shape
        torch.Size([3, 4])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(grady_x,2*x*z**2)
        True
        >>> torch.allclose(grady_z,x**2*2*z)
        True
        
        >>> f = lambda x,z: (x**2*z**2).sum((-2,-1))
        >>> x = torch.rand(3,4,5,generator=rng) 
        >>> z = torch.rand(3,4,5,generator=rng) 
        >>> (grady_x,),y = gradb(f,x,bkwargs={"z":z},bdims=1)
        >>> grady_x.shape 
        torch.Size([3, 4, 5])
        >>> y.shape
        torch.Size([3])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(grady_x,2*x*z**2)
        True
    """
    if isinstance(x,torch.Tensor): x = (x,)
    lenx = len(x)
    assert len(x)>=1
    batch_shape = list(x[0].shape[:bdims])
    for i,xi in enumerate(x): assert list(xi.shape[:bdims])==batch_shape, "input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for k,v in bkwargs.items(): assert list(v.shape[:bdims])==batch_shape, "bkwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_bkwargs = len(bkwargs) 
    bkwargs_keys = list(bkwargs.keys())
    def fwrap(*inputs):
        x = inputs[:lenx]
        bkwargs_vals = inputs[lenx:(lenx+len_bkwargs)]
        bkwargs = {bkwargs_keys[l]:bkwargs_vals[l] for l in range(len_bkwargs)}
        y = f(*x,**bkwargs)
        return y,y
    gradfwrap = torch.func.grad(fwrap,argnums=tuple(i for i in range(len(x))),has_aux=True)
    gradfwrapvec = torch.vmap(gradfwrap,in_dims=(0,)*(lenx+len_bkwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=bdims-1) if bdims>0 else xi[None,...] for xi in x]
    bkwargs_vals_input = [bkwargs[key].flatten(end_dim=bdims-1) if bdims>0 else bkwargs[key][None,...] for key in bkwargs_keys]
    gradys,y = gradfwrapvec(*x_input,*bkwargs_vals_input)
    if bdims==0:
        return tuple(grady[0] for grady in gradys),y[0]
    else:
        return tuple(grady.reshape(batch_shape+list(grady.shape[1:])) for grady in gradys),y.reshape(batch_shape+list(y.shape[1:]))

def jacfwdb(f, x, bkwargs={}, bdims=0, chunk_size=None):
    r"""
    Batched `torch.func.jacfwd` with function evaluation 

    Args:
        f (callable): Function to compute `torch.func.jacfwd` of. 
        x (Union[torch.Tensor,Tuple]): (batched) `Torch.Tensor` items whose jacobians will be computed.
        bkwargs (dict): (batched) `Torch.Tensor` items whose jacobians will not be computed.
        bdims (int): number of batch dimensions 
        chunk_size (int): to be passed into `torch.func.vmap`. 
    
    Returns: 
        jacys (tuple): `torch.Tensor` jacobians, one with respect to each item in `x`. 
        y (torch.Tensor): Function evaluations. 
    
    Examples: 
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

        >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        >>> x = torch.rand(5,generator=rng) 
        >>> (jacy,),y = jacfwdb(f,x)
        >>> jacy.shape 
        torch.Size([3, 5])
        >>> y.shape 
        torch.Size([3])
        >>> torch.allclose(y,f(x))
        True
        >>> torch.allclose(jacy,torch.arange(2,5)[:,None]*x**torch.arange(1,4)[:,None])
        True

        >>> f = lambda x,z: (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum(-3)
        >>> x = torch.rand(5,generator=rng) 
        >>> z = torch.rand(5,generator=rng) 
        >>> (jacy_x,jacy_z),y = jacfwdb(f,(x,z))
        >>> jacy_x.shape 
        torch.Size([2, 2, 5])
        >>> jacy_z.shape 
        torch.Size([2, 2, 5])
        >>> y.shape
        torch.Size([2, 2])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None]*x**torch.arange(1,3)[:,None,None]*z**torch.arange(3,5)[None,:,None])
        True
        >>> torch.allclose(jacy_z,x**torch.arange(2,4)[:,None,None]*torch.arange(3,5)[None,:,None]*z**torch.arange(2,4)[None,:,None])
        True
        
        >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        >>> x = torch.rand(3,4,5,generator=rng) 
        >>> (jacy,),y = jacfwdb(f,x,bdims=2)
        >>> jacy.shape 
        torch.Size([3, 4, 3, 5])
        >>> y.shape 
        torch.Size([3, 4, 3])
        >>> torch.allclose(y,f(x))
        True
        >>> torch.allclose(jacy,torch.arange(2,5)[:,None]*x[...,None,:]**torch.arange(1,4)[:,None])
        True

        >>> f = lambda x,z: (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum(-3)
        >>> x = torch.rand(3,4,5,generator=rng) 
        >>> z = torch.rand(3,4,5,generator=rng) 
        >>> (jacy_x,jacy_z),y = jacfwdb(f,(x,z),bdims=2)
        >>> jacy_x.shape 
        torch.Size([3, 4, 2, 2, 5])
        >>> jacy_z.shape 
        torch.Size([3, 4, 2, 2, 5])
        >>> y.shape
        torch.Size([3, 4, 2, 2])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None]*x[...,None,None,:]**torch.arange(1,3)[:,None,None]*z[...,None,None,:]**torch.arange(3,5)[None,:,None])
        True
        >>> torch.allclose(jacy_z,x[...,None,None,:]**torch.arange(2,4)[:,None,None]*torch.arange(3,5)[None,:,None]*z[...,None,None,:]**torch.arange(2,4)[None,:,None])
        True
        
        >>> f = lambda x,z: (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum((-4,-3))
        >>> x = torch.rand(3,4,5,6,generator=rng) 
        >>> z = torch.rand(3,4,5,6,generator=rng) 
        >>> (jacy_x,),y = jacfwdb(f,x,bkwargs={"z":z},bdims=2)
        >>> jacy_x.shape 
        torch.Size([3, 4, 2, 2, 5, 6])
        >>> y.shape
        torch.Size([3, 4, 2, 2])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None,None]*x[...,None,None,:,:]**torch.arange(1,3)[:,None,None,None]*z[...,None,None,:,:]**torch.arange(3,5)[None,:,None,None])
        True

        >>> def f(x):
        ...     y = (x[...,None]**torch.arange(2,4)).sum(-2)
        ...     u = (x[...,None]**torch.arange(3,5)).sum(-2)
        ...     v = (x[...,None]**torch.arange(4,6)).sum(-2)
        ...     return y,u,v
        >>> x = torch.rand(5,generator=rng) 
        >>> ((jacy_x,),(jacu_x,),(jacv_x,)),(y,u,v) = jacfwdb(f,x)
        >>> jacy_x.shape
        torch.Size([2, 5])
        >>> jacu_x.shape
        torch.Size([2, 5])
        >>> jacv_x.shape
        torch.Size([2, 5])
        >>> y.shape 
        torch.Size([2])
        >>> u.shape 
        torch.Size([2])
        >>> v.shape
        torch.Size([2])
        
        >>> def f(x,z):
        ...     y = (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum(-3)
        ...     u = (x[...,None,None]**torch.arange(3,5)[:,None]*z[...,None,None]**torch.arange(4,6)[None,:]).sum(-3)
        ...     v = (x[...,None,None]**torch.arange(4,6)[:,None]*z[...,None,None]**torch.arange(5,7)[None,:]).sum(-3)
        ...     return y,u,v
        >>> x = torch.rand(3,4,5,generator=rng) 
        >>> z = torch.rand(3,4,5,generator=rng) 
        >>> ((jacy_x,jacy_z),(jacu_x,jacu_z),(jacv_x,jacv_z)),(y,u,v) = jacfwdb(f,(x,z),bdims=2)
        >>> jacy_x.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> jacy_z.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> jacu_x.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> jacu_z.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> jacv_x.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> jacv_z.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> y.shape 
        torch.Size([3, 4, 2, 2])
        >>> u.shape 
        torch.Size([3, 4, 2, 2])
        >>> v.shape
        torch.Size([3, 4, 2, 2])
    """
    if isinstance(x,torch.Tensor): x = (x,)
    lenx = len(x) 
    assert len(x)>=1
    batch_shape = list(x[0].shape[:bdims])
    for i,xi in enumerate(x): assert list(xi.shape[:bdims])==batch_shape, "input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for k,v in bkwargs.items(): assert list(v.shape[:bdims])==batch_shape, "bkwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_bkwargs = len(bkwargs) 
    bkwargs_keys = list(bkwargs.keys())
    def fwrap(*inputs):
        x = inputs[:lenx]
        bkwargs_vals = inputs[lenx:(lenx+len_bkwargs)]
        bkwargs = {bkwargs_keys[l]:bkwargs_vals[l] for l in range(len_bkwargs)}
        y = f(*x,**bkwargs)
        return y,y
    jacfwrap = torch.func.jacfwd(fwrap,argnums=tuple(i for i in range(len(x))),has_aux=True)
    jacfwrapvec = torch.vmap(jacfwrap,in_dims=(0,)*(lenx+len_bkwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=bdims-1) if bdims>0 else xi[None,...] for xi in x]
    bkwargs_vals_input = [bkwargs[key].flatten(end_dim=bdims-1) if bdims>0 else bkwargs[key][None,...] for key in bkwargs_keys]
    jacys,y = jacfwrapvec(*x_input,*bkwargs_vals_input)
    if isinstance(y,torch.Tensor):
        if bdims==0:
            return tuple(jacy[0] for jacy in jacys),y[0]
        else:
            return tuple(jacy.reshape(batch_shape+list(jacy.shape[1:])) for jacy in jacys),y.reshape(batch_shape+list(y.shape[1:]))
    else:
        if bdims==0:
            return tuple(tuple(jacyk[0] for jacyk in jacy) for jacy in jacys),tuple(yk[0] for yk in y)
        else:
            return tuple(tuple(jacyk.reshape(batch_shape+list(jacyk.shape[1:])) for jacyk in jacy) for jacy in jacys),tuple(yk.reshape(batch_shape+list(yk.shape[1:])) for yk in y)

def jacrevb(f, x, bkwargs={}, bdims=0, chunk_size=None):
    r"""
    Batched `torch.func.jacrev` with function evaluation 

    Args:
        f (callable): Function to compute `torch.func.jacrev` of. 
        x (Tuple): (Union[torch.Tensor,Tuple]) `Torch.Tensor` items whose jacobians will be computed.
        bkwargs (dict): (batched) `Torch.Tensor` items whose jacobians will not be computed.
        bdims (int): number of batch dimensions 
        chunk_size (int): to be passed into `torch.func.vmap`. 
    
    Returns: 
        jacys (tuple): `torch.Tensor` jacobians, one with respect to each item in `x`. 
        y (torch.Tensor): Function evaluations. 
    
    Examples: 
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

        >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        >>> x = torch.rand(5,generator=rng) 
        >>> (jacy,),y = jacrevb(f,x)
        >>> jacy.shape 
        torch.Size([3, 5])
        >>> y.shape 
        torch.Size([3])
        >>> torch.allclose(y,f(x))
        True
        >>> torch.allclose(jacy,torch.arange(2,5)[:,None]*x**torch.arange(1,4)[:,None])
        True

        >>> f = lambda x,z: (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum(-3)
        >>> x = torch.rand(5,generator=rng) 
        >>> z = torch.rand(5,generator=rng) 
        >>> (jacy_x,jacy_z),y = jacrevb(f,(x,z))
        >>> jacy_x.shape 
        torch.Size([2, 2, 5])
        >>> jacy_z.shape 
        torch.Size([2, 2, 5])
        >>> y.shape
        torch.Size([2, 2])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None]*x**torch.arange(1,3)[:,None,None]*z**torch.arange(3,5)[None,:,None])
        True
        >>> torch.allclose(jacy_z,x**torch.arange(2,4)[:,None,None]*torch.arange(3,5)[None,:,None]*z**torch.arange(2,4)[None,:,None])
        True
        
        >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        >>> x = torch.rand(3,4,5,generator=rng) 
        >>> (jacy,),y = jacrevb(f,x,bdims=2)
        >>> jacy.shape 
        torch.Size([3, 4, 3, 5])
        >>> y.shape 
        torch.Size([3, 4, 3])
        >>> torch.allclose(y,f(x))
        True
        >>> torch.allclose(jacy,torch.arange(2,5)[:,None]*x[...,None,:]**torch.arange(1,4)[:,None])
        True

        >>> f = lambda x,z: (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum(-3)
        >>> x = torch.rand(3,4,5,generator=rng) 
        >>> z = torch.rand(3,4,5,generator=rng) 
        >>> (jacy_x,jacy_z),y = jacrevb(f,(x,z),bdims=2)
        >>> jacy_x.shape 
        torch.Size([3, 4, 2, 2, 5])
        >>> jacy_z.shape 
        torch.Size([3, 4, 2, 2, 5])
        >>> y.shape
        torch.Size([3, 4, 2, 2])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None]*x[...,None,None,:]**torch.arange(1,3)[:,None,None]*z[...,None,None,:]**torch.arange(3,5)[None,:,None])
        True
        >>> torch.allclose(jacy_z,x[...,None,None,:]**torch.arange(2,4)[:,None,None]*torch.arange(3,5)[None,:,None]*z[...,None,None,:]**torch.arange(2,4)[None,:,None])
        True
        
        >>> f = lambda x,z: (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum((-4,-3))
        >>> x = torch.rand(3,4,5,6,generator=rng) 
        >>> z = torch.rand(3,4,5,6,generator=rng) 
        >>> (jacy_x,),y = jacrevb(f,x,bkwargs={"z":z},bdims=2)
        >>> jacy_x.shape 
        torch.Size([3, 4, 2, 2, 5, 6])
        >>> y.shape
        torch.Size([3, 4, 2, 2])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None,None]*x[...,None,None,:,:]**torch.arange(1,3)[:,None,None,None]*z[...,None,None,:,:]**torch.arange(3,5)[None,:,None,None])
        True

        >>> def f(x):
        ...     y = (x[...,None]**torch.arange(2,4)).sum(-2)
        ...     u = (x[...,None]**torch.arange(3,5)).sum(-2)
        ...     v = (x[...,None]**torch.arange(4,6)).sum(-2)
        ...     return y,u,v
        >>> x = torch.rand(5,generator=rng) 
        >>> ((jacy_x,),(jacu_x,),(jacv_x,)),(y,u,v) = jacrevb(f,x)
        >>> jacy_x.shape
        torch.Size([2, 5])
        >>> jacu_x.shape
        torch.Size([2, 5])
        >>> jacv_x.shape
        torch.Size([2, 5])
        >>> y.shape 
        torch.Size([2])
        >>> u.shape 
        torch.Size([2])
        >>> v.shape
        torch.Size([2])
        
        >>> def f(x,z):
        ...     y = (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum(-3)
        ...     u = (x[...,None,None]**torch.arange(3,5)[:,None]*z[...,None,None]**torch.arange(4,6)[None,:]).sum(-3)
        ...     v = (x[...,None,None]**torch.arange(4,6)[:,None]*z[...,None,None]**torch.arange(5,7)[None,:]).sum(-3)
        ...     return y,u,v
        >>> x = torch.rand(3,4,5,generator=rng) 
        >>> z = torch.rand(3,4,5,generator=rng) 
        >>> ((jacy_x,jacy_z),(jacu_x,jacu_z),(jacv_x,jacv_z)),(y,u,v) = jacrevb(f,(x,z),bdims=2)
        >>> jacy_x.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> jacy_z.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> jacu_x.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> jacu_z.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> jacv_x.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> jacv_z.shape
        torch.Size([3, 4, 2, 2, 5])
        >>> y.shape 
        torch.Size([3, 4, 2, 2])
        >>> u.shape 
        torch.Size([3, 4, 2, 2])
        >>> v.shape
        torch.Size([3, 4, 2, 2])
    """
    if isinstance(x,torch.Tensor): x = (x,)
    lenx = len(x) 
    assert len(x)>=1
    batch_shape = list(x[0].shape[:bdims])
    for i,xi in enumerate(x): assert list(xi.shape[:bdims])==batch_shape, "input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for k,v in bkwargs.items(): assert list(v.shape[:bdims])==batch_shape, "bkwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_bkwargs = len(bkwargs) 
    bkwargs_keys = list(bkwargs.keys())
    def fwrap(*inputs):
        x = inputs[:lenx]
        bkwargs_vals = inputs[lenx:(lenx+len_bkwargs)]
        bkwargs = {bkwargs_keys[l]:bkwargs_vals[l] for l in range(len_bkwargs)}
        y = f(*x,**bkwargs)
        return y,y
    jacfwrap = torch.func.jacrev(fwrap,argnums=tuple(i for i in range(len(x))),has_aux=True)
    jacfwrapvec = torch.vmap(jacfwrap,in_dims=(0,)*(lenx+len_bkwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=bdims-1) if bdims>0 else xi[None,...] for xi in x]
    bkwargs_vals_input = [bkwargs[key].flatten(end_dim=bdims-1) if bdims>0 else bkwargs[key][None,...] for key in bkwargs_keys]
    jacys,y = jacfwrapvec(*x_input,*bkwargs_vals_input)
    if isinstance(y,torch.Tensor):
        if bdims==0:
            return tuple(jacy[0] for jacy in jacys),y[0]
        else:
            return tuple(jacy.reshape(batch_shape+list(jacy.shape[1:])) for jacy in jacys),y.reshape(batch_shape+list(y.shape[1:]))
    else:
        if bdims==0:
            return tuple(tuple(jacyk[0] for jacyk in jacy) for jacy in jacys),tuple(yk[0] for yk in y)
        else:
            return tuple(tuple(jacyk.reshape(batch_shape+list(jacyk.shape[1:])) for jacyk in jacy) for jacy in jacys),tuple(yk.reshape(batch_shape+list(yk.shape[1:])) for yk in y)

def jvpb(f, x, p, bkwargs={}, bdims=0, chunk_size=None):
    r"""
    Batched `torch.func.jvp` with function evaluation (forward-mode auto-diff)

    Args:
        f (callable): Function to compute `torch.func.jvp` of.
        x (Union[torch.Tensor,Tuple]): (batched) `Torch.Tensor` primals.
        p (Union[torch.Tensor,Tuple]): (batched) `Torch.Tensor` tangents.
        bkwargs (dict): (batched) `Torch.Tensor` items whose jacobians will not be computed.
        bdims (int): number of batch dimensions 
        chunk_size (int): to be passed into `torch.func.vmap`. 
    
    Returns: 
        jvp (tuple): `torch.Tensor` jacobian vector products, one with respect to each item in `x`. 
        y (torch.Tensor): Function evaluations. 
    
    Examples: 
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

        >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        >>> x = torch.rand(5,generator=rng)
        >>> p = torch.rand(5,generator=rng)
        >>> (jvpy,),y = jvpb(f,x,p)
        >>> jvpy.shape 
        torch.Size([3])
        >>> y.shape 
        torch.Size([3])
        >>> torch.allclose(y,f(x))
        True
        >>> (jac_x,),_ = jacfwdb(f,x)
        >>> torch.allclose(jvpy,jac_x@p)
        True

        >>> f = lambda x,z: (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        >>> x = torch.rand(5,generator=rng)
        >>> z = torch.rand(4,generator=rng)
        >>> p = torch.rand(5,generator=rng)
        >>> q = torch.rand(4,generator=rng)
        >>> (jvpy,),y = jvpb(f,(x,z),(p,q))
        >>> jvpy.shape
        torch.Size([3])
        >>> y.shape
        torch.Size([3])
        >>> torch.allclose(y,f(x,z))
        True
        >>> (jac_x,jac_z),_ = jacfwdb(f,(x,z))
        >>> torch.allclose(jvpy,jac_x@p+jac_z@q)
        True
        
        >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        >>> x = torch.rand(6,4,5,generator=rng)
        >>> p = torch.rand(6,4,5,generator=rng)
        >>> (jvpy,),y = jvpb(f,x,p,bdims=2)
        >>> jvpy.shape 
        torch.Size([6, 4, 3])
        >>> y.shape 
        torch.Size([6, 4, 3])
        >>> torch.allclose(y,f(x))
        True
        >>> (jac_x,),_ = jacfwdb(f,x,bdims=2)
        >>> torch.allclose(jvpy,(jac_x*p[...,None,:]).sum(-1))
        True

        >>> f = lambda x,z: (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        >>> x = torch.rand(6,7,5,generator=rng)
        >>> z = torch.rand(6,7,4,generator=rng)
        >>> p = torch.rand(6,7,5,generator=rng)
        >>> q = torch.rand(6,7,4,generator=rng)
        >>> (jvpy,),y = jvpb(f,(x,z),(p,q),bdims=2)
        >>> jvpy.shape
        torch.Size([6, 7, 3])
        >>> y.shape
        torch.Size([6, 7, 3])
        >>> torch.allclose(y,f(x,z))
        True
        >>> (jac_x,jac_z),_ = jacfwdb(f,(x,z),bdims=2)
        >>> torch.allclose(jvpy,(jac_x*p[...,None,:]).sum(-1)+(jac_z*q[...,None,:]).sum(-1))
        True
        
        >>> f = lambda x,z: (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        >>> x = torch.rand(6,7,5,generator=rng)
        >>> z = torch.rand(6,7,4,generator=rng)
        >>> p = torch.rand(6,7,5,generator=rng)
        >>> (jvpy,),y = jvpb(f,x,p,bkwargs={"z":z},bdims=2)
        >>> jvpy.shape
        torch.Size([6, 7, 3])
        >>> y.shape
        torch.Size([6, 7, 3])
        >>> torch.allclose(y,f(x,z))
        True
        >>> (jac_x,jac_z),_ = jacfwdb(f,(x,z),bdims=2)
        >>> torch.allclose(jvpy,(jac_x*p[...,None,:]).sum(-1))
        True
        
        >>> def f(x):
        ...     y = (x[...,:,None]**torch.arange(2,5)).sum(-2)
        ...     u = (x[...,:,None]**torch.arange(3,6)).sum(-2)
        ...     v = (x[...,:,None]**torch.arange(4,7)).sum(-2)
        ...     return y,u,v
        >>> x = torch.rand(5,generator=rng)
        >>> p = torch.rand(5,generator=rng)
        >>> (jvpy,jvpu,jvpv),(y,u,v) = jvpb(f,x,p)
        >>> jvpy.shape
        torch.Size([3])
        >>> jvpu.shape
        torch.Size([3])
        >>> jvpv.shape
        torch.Size([3])
        >>> y.shape
        torch.Size([3])
        >>> u.shape
        torch.Size([3])
        >>> v.shape
        torch.Size([3])
        >>> torch.allclose(y,f(x)[0])
        True
        >>> torch.allclose(u,f(x)[1])
        True
        >>> torch.allclose(v,f(x)[2])
        True
        >>> ((jacy_x,),(jacu_x,),(jacv_x,)),_ = jacfwdb(f,x)
        >>> torch.allclose(jvpy,(jacy_x*p[...,None,:]).sum(-1))
        True
        >>> torch.allclose(jvpu,(jacu_x*p[...,None,:]).sum(-1))
        True
        >>> torch.allclose(jvpv,(jacv_x*p[...,None,:]).sum(-1))
        True
        
        >>> def f(x,z):
        ...     y = (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        ...     u = (x[...,:,None,None]**torch.arange(3,6)*z[...,None,:,None]**torch.arange(2,5)).sum((-3,-2))
        ...     v = (x[...,:,None,None]**torch.arange(4,7)*z[...,None,:,None]**torch.arange(3,6)).sum((-3,-2))
        ...     return y,u,v
        >>> x = torch.rand(6,7,5,generator=rng)
        >>> z = torch.rand(6,7,4,generator=rng)
        >>> p = torch.rand(6,7,5,generator=rng)
        >>> q = torch.rand(6,7,4,generator=rng)
        >>> (jvpy,jvpu,jvpv),(y,u,v) = jvpb(f,(x,z),(p,q),bdims=2)
        >>> jvpy.shape
        torch.Size([6, 7, 3])
        >>> jvpu.shape
        torch.Size([6, 7, 3])
        >>> jvpv.shape
        torch.Size([6, 7, 3])
        >>> y.shape
        torch.Size([6, 7, 3])
        >>> u.shape
        torch.Size([6, 7, 3])
        >>> v.shape
        torch.Size([6, 7, 3])
        >>> torch.allclose(y,f(x,z)[0])
        True
        >>> torch.allclose(u,f(x,z)[1])
        True
        >>> torch.allclose(v,f(x,z)[2])
        True
        >>> ((jacy_x,jacy_z),(jacu_x,jacu_z),(jacv_x,jacv_z)),_ = jacfwdb(f,(x,z),bdims=2)
        >>> torch.allclose(jvpy,(jacy_x*p[...,None,:]).sum(-1)+(jacy_z*q[...,None,:]).sum(-1))
        True
        >>> torch.allclose(jvpu,(jacu_x*p[...,None,:]).sum(-1)+(jacu_z*q[...,None,:]).sum(-1))
        True
        >>> torch.allclose(jvpv,(jacv_x*p[...,None,:]).sum(-1)+(jacv_z*q[...,None,:]).sum(-1))
        True
    """
    if isinstance(x,torch.Tensor): x = (x,)
    if isinstance(p,torch.Tensor): p = (p,)
    lenx = len(x) 
    assert len(x)>=1
    assert len(p)==lenx
    batch_shape = list(x[0].shape[:bdims])
    for i,xi in enumerate(x): assert list(xi.shape[:bdims])==batch_shape, "x input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for i,pi in enumerate(p): assert list(pi.shape[:bdims])==batch_shape, "p input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for k,v in bkwargs.items(): assert list(v.shape[:bdims])==batch_shape, "bkwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_bkwargs = len(bkwargs) 
    bkwargs_keys = list(bkwargs.keys())
    def jvpfwrap(*inputs):
        x,p,bkwargs_vals = inputs[:lenx],inputs[lenx:(2*lenx)],inputs[(2*lenx):]
        return torch.func.jvp(lambda *x: f(*x,*bkwargs_vals),x,p)
    jvpfwrapvec = torch.vmap(jvpfwrap,in_dims=(0,)*(2*lenx+len_bkwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=bdims-1) if bdims>0 else xi[None,...] for xi in x]
    p_input = [pi.flatten(end_dim=bdims-1) if bdims>0 else pi[None,...] for pi in p]
    bkwargs_vals_input = [bkwargs[key].flatten(end_dim=bdims-1) if bdims>0 else bkwargs[key][None,...] for key in bkwargs_keys]
    y,jvpy = jvpfwrapvec(*x_input,*p_input,*bkwargs_vals_input)
    if isinstance(jvpy,torch.Tensor): jvpy = (jvpy,)
    if isinstance(y,torch.Tensor):
        if bdims==0:
            return tuple(jvpyk[0] for jvpyk in jvpy),y[0]
        else:
            return tuple(jvpyk.reshape(batch_shape+list(jvpyk.shape[1:])) for jvpyk in jvpy),y.reshape(batch_shape+list(y.shape[1:]))
    else:
        if bdims==0:
            return tuple(jvpyk[0] for jvpyk in jvpy),tuple(yk[0] for yk in y)
        else:
            return tuple(jvpyk.reshape(batch_shape+list(jvpyk.shape[1:])) for jvpyk in jvpy),tuple(yk.reshape(batch_shape+list(yk.shape[1:])) for yk in y)

def vjpb(f, x, p, bkwargs={}, bdims=0, chunk_size=None):
    r"""
    Batched `torch.func.vjp` with function evaluation (forward-mode auto-diff)

    Args:
        f (callable): Function to compute `torch.func.vjp` of.
        x (Union[torch.Tensor,Tuple]): (batched) `Torch.Tensor` primals.
        p (Union[torch.Tensor,Tuple]): (batched) `Torch.Tensor` tangents.
        bkwargs (dict): (batched) `Torch.Tensor` items whose jacobians will not be computed.
        bdims (int): number of batch dimensions 
        chunk_size (int): to be passed into `torch.func.vmap`. 
    
    Returns: 
        vjp (tuple): `torch.Tensor` jacobian vector products, one with respect to each item in `x`. 
        y (torch.Tensor): Function evaluations. 
    
    Examples: 
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

        >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        >>> x = torch.rand(5,generator=rng)
        >>> p = torch.rand(3,generator=rng)
        >>> (vjpx,),y = vjpb(f,x,p)
        >>> vjpx.shape 
        torch.Size([5])
        >>> y.shape 
        torch.Size([3])
        >>> torch.allclose(y,f(x))
        True
        >>> (jac_x,),_ = jacfwdb(f,x)
        >>> torch.allclose(vjpx,p@jac_x)
        True

        >>> def f(x,z):
        ...     y = (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        ...     u = (x[...,:,None,None]**torch.arange(3,6)*z[...,None,:,None]**torch.arange(2,5)).sum((-3,-2))
        ...     v = (x[...,:,None,None]**torch.arange(4,7)*z[...,None,:,None]**torch.arange(3,6)).sum((-3,-2))
        ...     return y,u,v
        >>> x = torch.rand(5,generator=rng)
        >>> z = torch.rand(6,generator=rng)
        >>> p = torch.rand(3,generator=rng)
        >>> q = torch.rand(3,generator=rng)
        >>> r = torch.rand(3,generator=rng)
        >>> (vjp_x,vjp_z),(y,u,v) = vjpb(f,(x,z),(p,q,r))
        >>> vjp_x.shape
        torch.Size([5])
        >>> vjp_z.shape
        torch.Size([6])
        >>> y.shape
        torch.Size([3])
        >>> v.shape
        torch.Size([3])
        >>> torch.allclose(y,f(x,z)[0])
        True
        >>> torch.allclose(u,f(x,z)[1])
        True
        >>> torch.allclose(v,f(x,z)[1])
        False
        >>> ((jacy_x,jacy_z),(jacu_x,jacu_z),(jacv_x,jacv_z)),_ = jacfwdb(f,(x,z))
        >>> torch.allclose(vjp_x,p@jacy_x+q@jacu_x+r@jacv_x)
        True
        >>> torch.allclose(vjp_z,p@jacy_z+q@jacu_z+r@jacv_z)
        True
        
        >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        >>> x = torch.rand(7,5,generator=rng)
        >>> p = torch.rand(7,3,generator=rng)
        >>> (vjpx,),y = vjpb(f,x,p,bdims=1)
        >>> vjpx.shape 
        torch.Size([7, 5])
        >>> y.shape 
        torch.Size([7, 3])
        >>> torch.allclose(y,f(x))
        True
        >>> (jac_x,),_ = jacfwdb(f,x,bdims=1)
        >>> torch.allclose(vjpx,(p[...,None]*jac_x).sum(-2))
        True

        >>> def f(x,z):
        ...     y = (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        ...     u = (x[...,:,None,None]**torch.arange(3,6)*z[...,None,:,None]**torch.arange(2,5)).sum((-3,-2))
        ...     v = (x[...,:,None,None]**torch.arange(4,7)*z[...,None,:,None]**torch.arange(3,6)).sum((-3,-2))
        ...     return y,u,v
        >>> x = torch.rand(7,5,generator=rng)
        >>> z = torch.rand(7,6,generator=rng)
        >>> p = torch.rand(7,3,generator=rng)
        >>> q = torch.rand(7,3,generator=rng)
        >>> r = torch.rand(7,3,generator=rng)
        >>> (vjp_x,vjp_z),(y,u,v) = vjpb(f,(x,z),(p,q,r),bdims=1)
        >>> vjp_x.shape
        torch.Size([7, 5])
        >>> vjp_z.shape
        torch.Size([7, 6])
        >>> y.shape
        torch.Size([7, 3])
        >>> v.shape
        torch.Size([7, 3])
        >>> torch.allclose(y,f(x,z)[0])
        True
        >>> torch.allclose(u,f(x,z)[1])
        True
        >>> torch.allclose(v,f(x,z)[1])
        False
        >>> ((jacy_x,jacy_z),(jacu_x,jacu_z),(jacv_x,jacv_z)),_ = jacfwdb(f,(x,z),bdims=1)
        >>> torch.allclose(vjp_x,(p[...,None]*jacy_x).sum(-2)+(q[...,None]*jacu_x).sum(-2)+(r[...,None]*jacv_x).sum(-2))
        True
        >>> torch.allclose(vjp_z,(p[...,None]*jacy_z).sum(-2)+(q[...,None]*jacu_z).sum(-2)+(r[...,None]*jacv_z).sum(-2))
        True

        >>> def f(x,z):
        ...     y = (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        ...     u = (x[...,:,None,None]**torch.arange(3,6)*z[...,None,:,None]**torch.arange(2,5)).sum((-3,-2))
        ...     v = (x[...,:,None,None]**torch.arange(4,7)*z[...,None,:,None]**torch.arange(3,6)).sum((-3,-2))
        ...     return y,u,v
        >>> x = torch.rand(7,5,generator=rng)
        >>> z = torch.rand(7,6,generator=rng)
        >>> p = torch.rand(7,3,generator=rng)
        >>> q = torch.rand(7,3,generator=rng)
        >>> r = torch.rand(7,3,generator=rng)
        >>> (vjp_x,),(y,u,v) = vjpb(f,x,(p,q,r),bkwargs={"z":z},bdims=1)
        >>> vjp_x.shape
        torch.Size([7, 5])
        >>> y.shape
        torch.Size([7, 3])
        >>> v.shape
        torch.Size([7, 3])
        >>> torch.allclose(y,f(x,z)[0])
        True
        >>> torch.allclose(u,f(x,z)[1])
        True
        >>> torch.allclose(v,f(x,z)[1])
        False
        >>> ((jacy_x,jacy_z),(jacu_x,jacu_z),(jacv_x,jacv_z)),_ = jacfwdb(f,(x,z),bdims=1)
        >>> torch.allclose(vjp_x,(p[...,None]*jacy_x).sum(-2)+(q[...,None]*jacu_x).sum(-2)+(r[...,None]*jacv_x).sum(-2))
        True
    """
    if isinstance(x,torch.Tensor): x = (x,)
    if isinstance(p,torch.Tensor): p = (p,)
    lenx = len(x) 
    assert len(x)>=1
    lenp = len(p)
    assert len(p)>=1
    batch_shape = list(x[0].shape[:bdims])
    for i,xi in enumerate(x): assert list(xi.shape[:bdims])==batch_shape, "x input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for i,pi in enumerate(p): assert list(pi.shape[:bdims])==batch_shape, "p input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for k,v in bkwargs.items(): assert list(v.shape[:bdims])==batch_shape, "bkwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_bkwargs = len(bkwargs) 
    bkwargs_keys = list(bkwargs.keys())
    def vjpfwrap(*inputs):
        x,p,bkwargs_vals = inputs[:lenx],inputs[lenx:(lenx+lenp)],inputs[(lenx+lenp):]
        y,vjpfwrap_inner = torch.func.vjp(lambda *x: f(*x,*bkwargs_vals),*x)
        vjpywrap = vjpfwrap_inner(p[0] if isinstance(y,torch.Tensor) else p)
        return y,vjpywrap
    vjpfwrapvec = torch.vmap(vjpfwrap,in_dims=(0,)*(lenx+lenp+len_bkwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=bdims-1) if bdims>0 else xi[None,...] for xi in x]
    p_input = [pi.flatten(end_dim=bdims-1) if bdims>0 else pi[None,...] for pi in p]
    bkwargs_vals_input = [bkwargs[key].flatten(end_dim=bdims-1) if bdims>0 else bkwargs[key][None,...] for key in bkwargs_keys]
    y,vjpys = vjpfwrapvec(*x_input,*p_input,*bkwargs_vals_input)
    if isinstance(y,torch.Tensor):
        if bdims==0:
            return (vjpy[0] for vjpy in vjpys),y[0]
        else:
            return (vjpy.reshape(batch_shape+list(vjpy.shape[1:])) for vjpy in vjpys),y.reshape(batch_shape+list(y.shape[1:]))
    else:
        if bdims==0:
            return (vjpy[0] for vjpy in vjpys),(yk[0] for yk in y)
        else:
            return (vjpy.reshape(batch_shape+list(vjpy.shape[1:])) for vjpy in vjpys),(yk.reshape(batch_shape+list(yk.shape[1:])) for yk in y)
