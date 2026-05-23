import torch 

def gradb(f, *x, batch_kwargs={}, batchdims=0, chunk_size=None):
    r"""
    Batched `torch.func.grad` and function evaluation 

    Args:
        f (callable): Function to compute `torch.func.grad` of. 
        x (Tuple): (batched) `Torch.Tensor` items whose gradients will be computed.
        batch_kwargs (dict): (batched) `Torch.Tensor` items whose gradients will not be computed.
        batchdims (int): number of batch dimensions 
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
        >>> (grady_x,grady_z),y = gradb(f,x,z)
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
        >>> (grady,),y = gradb(f,x,batchdims=2)
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
        >>> (grady_x,grady_z),y = gradb(f,x,z,batchdims=2)
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
        >>> (grady_x,),y = gradb(f,x,batch_kwargs={"z":z},batchdims=1)
        >>> grady_x.shape 
        torch.Size([3, 4, 5])
        >>> y.shape
        torch.Size([3])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(grady_x,2*x*z**2)
        True
    """
    lenx = len(x) 
    assert len(x)>=1
    batch_shape = list(x[0].shape[:batchdims])
    for i,xi in enumerate(x): assert list(xi.shape[:batchdims])==batch_shape, "input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for k,v in batch_kwargs.items(): assert list(v.shape[:batchdims])==batch_shape, "batch_kwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_batch_kwargs = len(batch_kwargs) 
    batch_kwargs_keys = list(batch_kwargs.keys())
    def fwrap(*inputs):
        x = inputs[:lenx]
        batch_kwargs_vals = inputs[lenx:(lenx+len_batch_kwargs)]
        batch_kwargs = {batch_kwargs_keys[l]:batch_kwargs_vals[l] for l in range(len_batch_kwargs)}
        y = f(*x,**batch_kwargs)
        return y,y
    gradfwrap = torch.func.grad(fwrap,argnums=tuple(i for i in range(len(x))),has_aux=True)
    gradfwrapvec = torch.vmap(gradfwrap,in_dims=(0,)*(lenx+len_batch_kwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=batchdims-1) if batchdims>0 else xi[None,...] for xi in x]
    batch_kwargs_vals_input = [batch_kwargs[key].flatten(end_dim=batchdims-1) if batchdims>0 else batch_kwargs[key][None,...] for key in batch_kwargs_keys]
    gradys,y = gradfwrapvec(*x_input,*batch_kwargs_vals_input)
    if batchdims==0:
        return tuple(grady[0] for grady in gradys),y[0]
    else:
        return tuple(grady.reshape(batch_shape+list(grady.shape[1:])) for grady in gradys),y.reshape(batch_shape+list(y.shape[1:]))

def jacfwdb(f, *x, batch_kwargs={}, batchdims=0, chunk_size=None):
    r"""
    Batched `torch.func.jacfwd` with function evaluation 

    Args:
        f (callable): Function to compute `torch.func.jacfwd` of. 
        x (Tuple): (batched) `Torch.Tensor` items whose jacobians will be computed.
        batch_kwargs (dict): (batched) `Torch.Tensor` items whose jacobians will not be computed.
        batchdims (int): number of batch dimensions 
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
        >>> (jacy_x,jacy_z),y = jacfwdb(f,x,z)
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
        >>> (jacy,),y = jacfwdb(f,x,batchdims=2)
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
        >>> (jacy_x,jacy_z),y = jacfwdb(f,x,z,batchdims=2)
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
        >>> (jacy_x,),y = jacfwdb(f,x,batch_kwargs={"z":z},batchdims=2)
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
        >>> ((jacy_x,jacy_z),(jacu_x,jacu_z),(jacv_x,jacv_z)),(y,u,v) = jacfwdb(f,x,z,batchdims=2)
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
    lenx = len(x) 
    assert len(x)>=1
    batch_shape = list(x[0].shape[:batchdims])
    for i,xi in enumerate(x): assert list(xi.shape[:batchdims])==batch_shape, "input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for k,v in batch_kwargs.items(): assert list(v.shape[:batchdims])==batch_shape, "batch_kwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_batch_kwargs = len(batch_kwargs) 
    batch_kwargs_keys = list(batch_kwargs.keys())
    def fwrap(*inputs):
        x = inputs[:lenx]
        batch_kwargs_vals = inputs[lenx:(lenx+len_batch_kwargs)]
        batch_kwargs = {batch_kwargs_keys[l]:batch_kwargs_vals[l] for l in range(len_batch_kwargs)}
        y = f(*x,**batch_kwargs)
        return y,y
    jacfwrap = torch.func.jacfwd(fwrap,argnums=tuple(i for i in range(len(x))),has_aux=True)
    jacfwrapvec = torch.vmap(jacfwrap,in_dims=(0,)*(lenx+len_batch_kwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=batchdims-1) if batchdims>0 else xi[None,...] for xi in x]
    batch_kwargs_vals_input = [batch_kwargs[key].flatten(end_dim=batchdims-1) if batchdims>0 else batch_kwargs[key][None,...] for key in batch_kwargs_keys]
    jacys,y = jacfwrapvec(*x_input,*batch_kwargs_vals_input)
    if isinstance(y,torch.Tensor):
        if batchdims==0:
            return tuple(jacy[0] for jacy in jacys),y[0]
        else:
            return tuple(jacy.reshape(batch_shape+list(jacy.shape[1:])) for jacy in jacys),y.reshape(batch_shape+list(y.shape[1:]))
    else:
        if batchdims==0:
            return tuple((jacyk[0] for jacyk in jacy) for jacy in jacys),tuple(yk[0] for yk in y)
        else:
            return tuple((jacyk.reshape(batch_shape+list(jacyk.shape[1:])) for jacyk in jacy) for jacy in jacys),tuple(yk.reshape(batch_shape+list(yk.shape[1:])) for yk in y)

def jacrevb(f, *x, batch_kwargs={}, batchdims=0, chunk_size=None):
    r"""
    Batched `torch.func.jacrev` with function evaluation 

    Args:
        f (callable): Function to compute `torch.func.jacrev` of. 
        x (Tuple): (batched) `Torch.Tensor` items whose jacobians will be computed.
        batch_kwargs (dict): (batched) `Torch.Tensor` items whose jacobians will not be computed.
        batchdims (int): number of batch dimensions 
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
        >>> (jacy_x,jacy_z),y = jacrevb(f,x,z)
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
        >>> (jacy,),y = jacrevb(f,x,batchdims=2)
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
        >>> (jacy_x,jacy_z),y = jacrevb(f,x,z,batchdims=2)
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
        >>> (jacy_x,),y = jacrevb(f,x,batch_kwargs={"z":z},batchdims=2)
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
        >>> ((jacy_x,jacy_z),(jacu_x,jacu_z),(jacv_x,jacv_z)),(y,u,v) = jacrevb(f,x,z,batchdims=2)
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
    lenx = len(x) 
    assert len(x)>=1
    batch_shape = list(x[0].shape[:batchdims])
    for i,xi in enumerate(x): assert list(xi.shape[:batchdims])==batch_shape, "input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for k,v in batch_kwargs.items(): assert list(v.shape[:batchdims])==batch_shape, "batch_kwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_batch_kwargs = len(batch_kwargs) 
    batch_kwargs_keys = list(batch_kwargs.keys())
    def fwrap(*inputs):
        x = inputs[:lenx]
        batch_kwargs_vals = inputs[lenx:(lenx+len_batch_kwargs)]
        batch_kwargs = {batch_kwargs_keys[l]:batch_kwargs_vals[l] for l in range(len_batch_kwargs)}
        y = f(*x,**batch_kwargs)
        return y,y
    jacfwrap = torch.func.jacrev(fwrap,argnums=tuple(i for i in range(len(x))),has_aux=True)
    jacfwrapvec = torch.vmap(jacfwrap,in_dims=(0,)*(lenx+len_batch_kwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=batchdims-1) if batchdims>0 else xi[None,...] for xi in x]
    batch_kwargs_vals_input = [batch_kwargs[key].flatten(end_dim=batchdims-1) if batchdims>0 else batch_kwargs[key][None,...] for key in batch_kwargs_keys]
    jacys,y = jacfwrapvec(*x_input,*batch_kwargs_vals_input)
    if isinstance(y,torch.Tensor):
        if batchdims==0:
            return tuple(jacy[0] for jacy in jacys),y[0]
        else:
            return tuple(jacy.reshape(batch_shape+list(jacy.shape[1:])) for jacy in jacys),y.reshape(batch_shape+list(y.shape[1:]))
    else:
        if batchdims==0:
            return tuple((jacyk[0] for jacyk in jacy) for jacy in jacys),tuple(yk[0] for yk in y)
        else:
            return tuple((jacyk.reshape(batch_shape+list(jacyk.shape[1:])) for jacyk in jacy) for jacy in jacys),tuple(yk.reshape(batch_shape+list(yk.shape[1:])) for yk in y)

def jvpb(f, x, p, batch_kwargs={}, batchdims=0, chunk_size=None):
    r"""
    Batched `torch.func.jvp` with function evaluation (forward-mode auto-diff)

    Args:
        f (callable): Function to compute `torch.func.jvp` of.
        x (Tuple): (batched) `Torch.Tensor` primals.
        p (Tuple): (batched) `Torch.Tensor` tangents.
        batch_kwargs (dict): (batched) `Torch.Tensor` items whose jacobians will not be computed.
        batchdims (int): number of batch dimensions 
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
        >>> jvpy,y = jvpb(f,(x,),(p,))
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
        >>> jvpy,y = jvpb(f,(x,z),(p,q))
        >>> jvpy.shape
        torch.Size([3])
        >>> y.shape
        torch.Size([3])
        >>> torch.allclose(y,f(x,z))
        True
        >>> (jac_x,jac_z),_ = jacfwdb(f,x,z)
        >>> torch.allclose(jvpy,jac_x@p+jac_z@q)
        True
        
        >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        >>> x = torch.rand(6,4,5,generator=rng)
        >>> p = torch.rand(6,4,5,generator=rng)
        >>> jvpy,y = jvpb(f,(x,),(p,),batchdims=2)
        >>> jvpy.shape 
        torch.Size([6, 4, 3])
        >>> y.shape 
        torch.Size([6, 4, 3])
        >>> torch.allclose(y,f(x))
        True
        >>> (jac_x,),_ = jacfwdb(f,x,batchdims=2)
        >>> torch.allclose(jvpy,(jac_x*p[...,None,:]).sum(-1))
        True

        >>> f = lambda x,z: (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        >>> x = torch.rand(6,7,5,generator=rng)
        >>> z = torch.rand(6,7,4,generator=rng)
        >>> p = torch.rand(6,7,5,generator=rng)
        >>> q = torch.rand(6,7,4,generator=rng)
        >>> jvpy,y = jvpb(f,(x,z),(p,q),batchdims=2)
        >>> jvpy.shape
        torch.Size([6, 7, 3])
        >>> y.shape
        torch.Size([6, 7, 3])
        >>> torch.allclose(y,f(x,z))
        True
        >>> (jac_x,jac_z),_ = jacfwdb(f,x,z,batchdims=2)
        >>> torch.allclose(jvpy,(jac_x*p[...,None,:]).sum(-1)+(jac_z*q[...,None,:]).sum(-1))
        True
        
        >>> f = lambda x,z: (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        >>> x = torch.rand(6,7,5,generator=rng)
        >>> z = torch.rand(6,7,4,generator=rng)
        >>> p = torch.rand(6,7,5,generator=rng)
        >>> jvpy,y = jvpb(f,(x,),(p,),batch_kwargs={"z":z},batchdims=2)
        >>> jvpy.shape
        torch.Size([6, 7, 3])
        >>> y.shape
        torch.Size([6, 7, 3])
        >>> torch.allclose(y,f(x,z))
        True
        >>> (jac_x,jac_z),_ = jacfwdb(f,x,z,batchdims=2)
        >>> torch.allclose(jvpy,(jac_x*p[...,None,:]).sum(-1))
        True
        
        >>> def f(x):
        ...     y = (x[...,:,None]**torch.arange(2,5)).sum(-2)
        ...     u = (x[...,:,None]**torch.arange(3,6)).sum(-2)
        ...     v = (x[...,:,None]**torch.arange(4,7)).sum(-2)
        ...     return y,u,v
        >>> x = torch.rand(5,generator=rng)
        >>> p = torch.rand(5,generator=rng)
        >>> (jvpy,jvpu,jvpv),(y,u,v) = jvpb(f,(x,),(p,))
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
        >>> torch.allclose(y,f(x,z)[0])
        False
        >>> torch.allclose(u,f(x,z)[1])
        True
        >>> torch.allclose(v,f(x,z)[2])
        True
        >>> ((jacy_x,),(jacu_x,),(jacv_x,)),_ = jacfwdb(f,x,z,batchdims=2)
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
        >>> (jvpy,jvpu,jvpv),(y,u,v) = jvpb(f,(x,z),(p,q),batchdims=2)
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
        >>> ((jacy_x,jacy_z),(jacu_x,jacu_z),(jacv_x,jacv_z)),_ = jacfwdb(f,x,z,batchdims=2)
        >>> torch.allclose(jvpy,(jacy_x*p[...,None,:]).sum(-1)+(jacy_z*q[...,None,:]).sum(-1))
        True
        >>> torch.allclose(jvpu,(jacu_x*p[...,None,:]).sum(-1)+(jacu_z*q[...,None,:]).sum(-1))
        True
        >>> torch.allclose(jvpv,(jacv_x*p[...,None,:]).sum(-1)+(jacv_z*q[...,None,:]).sum(-1))
        True
    """
    lenx = len(x) 
    assert len(x)>=1
    assert len(p)==lenx
    batch_shape = list(x[0].shape[:batchdims])
    for i,xi in enumerate(x): assert list(xi.shape[:batchdims])==batch_shape, "x input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for i,pi in enumerate(p): assert list(pi.shape[:batchdims])==batch_shape, "p input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for k,v in batch_kwargs.items(): assert list(v.shape[:batchdims])==batch_shape, "batch_kwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_batch_kwargs = len(batch_kwargs) 
    batch_kwargs_keys = list(batch_kwargs.keys())
    def jvpfwrap(*inputs):
        x,p,batch_kwargs_vals = inputs[:lenx],inputs[lenx:(2*lenx)],inputs[(2*lenx):]
        return torch.func.jvp(lambda *x: f(*x,*batch_kwargs_vals),x,p)
    jvpfwrapvec = torch.vmap(jvpfwrap,in_dims=(0,)*(2*lenx+len_batch_kwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=batchdims-1) if batchdims>0 else xi[None,...] for xi in x]
    p_input = [pi.flatten(end_dim=batchdims-1) if batchdims>0 else pi[None,...] for pi in p]
    batch_kwargs_vals_input = [batch_kwargs[key].flatten(end_dim=batchdims-1) if batchdims>0 else batch_kwargs[key][None,...] for key in batch_kwargs_keys]
    y,jvpy = jvpfwrapvec(*x_input,*p_input,*batch_kwargs_vals_input)
    if isinstance(y,torch.Tensor):
        if batchdims==0:
            return jvpy[0],y[0]
        else:
            return jvpy.reshape(batch_shape+list(jvpy.shape[1:])),y.reshape(batch_shape+list(y.shape[1:]))
    else:
        if batchdims==0:
            return tuple(jvpyk[0] for jvpyk in jvpy),tuple(yk[0] for yk in y)
        else:
            return tuple(jvpyk.reshape(batch_shape+list(jvpyk.shape[1:])) for jvpyk in jvpy),tuple(yk.reshape(batch_shape+list(yk.shape[1:])) for yk in y)

def vjpb(f, x, p, batch_kwargs={}, batchdims=0, chunk_size=None):
    r"""
    Batched `torch.func.vjp` with function evaluation (forward-mode auto-diff)

    Args:
        f (callable): Function to compute `torch.func.vjp` of.
        x (Tuple): (batched) `Torch.Tensor` primals.
        p (Tuple): (batched) `Torch.Tensor` tangents.
        batch_kwargs (dict): (batched) `Torch.Tensor` items whose jacobians will not be computed.
        batchdims (int): number of batch dimensions 
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
        >>> vjpy,y = vjpb(f,(x,),(p,))
        >>> vjpy.shape 
        torch.Size([5])
        >>> y.shape 
        torch.Size([3])
        >>> torch.allclose(y,f(x))
        True
        >>> (jac_x,),_ = jacfwdb(f,x)
        >>> torch.allclose(vjpy,p@jac_x)
        True

        # >>> f = lambda x,z: (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        # >>> x = torch.rand(5,generator=rng)
        # >>> z = torch.rand(4,generator=rng)
        # >>> p = torch.rand(5,generator=rng)
        # >>> q = torch.rand(4,generator=rng)
        # >>> vjpy,y = jvpb(f,(x,z),(p,q))
        # >>> vjpy.shape
        # torch.Size([3])
        # >>> y.shape
        # torch.Size([3])
        # >>> torch.allclose(y,f(x,z))
        # True
        # >>> jac_x,jac_z,_ = jacfwdb(f,x,z)
        # >>> torch.allclose(vjpy,p@jac_x+q@jac_z)
        # True
        
        # >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        # >>> x = torch.rand(6,4,5,generator=rng)
        # >>> p = torch.rand(6,4,5,generator=rng)
        # >>> jacy,y = jvpb(f,(x,),(p,),batchdims=2)
        # >>> jacy.shape 
        # torch.Size([6, 4, 3])
        # >>> y.shape 
        # torch.Size([6, 4, 3])
        # >>> torch.allclose(y,f(x))
        # True
        # >>> torch.allclose(jacy,(jacfwdb(f,x,batchdims=2)[0]*p[...,None,:]).sum(-1))
        # True

        # >>> f = lambda x,z: (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        # >>> x = torch.rand(6,7,5,generator=rng)
        # >>> z = torch.rand(6,7,4,generator=rng)
        # >>> p = torch.rand(6,7,5,generator=rng)
        # >>> q = torch.rand(6,7,4,generator=rng)
        # >>> jvpy,y = jvpb(f,(x,z),(p,q),batchdims=2)
        # >>> jvpy.shape
        # torch.Size([6, 7, 3])
        # >>> y.shape
        # torch.Size([6, 7, 3])
        # >>> torch.allclose(y,f(x,z))
        # True
        # >>> jac_x,jac_z,_ = jacfwdb(f,x,z,batchdims=2)
        # >>> torch.allclose(jvpy,(jac_x*p[...,None,:]).sum(-1)+(jac_z*q[...,None,:]).sum(-1))
        # True
        
        # >>> f = lambda x,z: (x[...,:,None,None]**torch.arange(2,5)*z[...,None,:,None]**torch.arange(1,4)).sum((-3,-2))
        # >>> x = torch.rand(6,7,5,generator=rng)
        # >>> z = torch.rand(6,7,4,generator=rng)
        # >>> p = torch.rand(6,7,5,generator=rng)
        # >>> jvpy,y = jvpb(f,(x,),(p,),batch_kwargs={"z":z},batchdims=2)
        # >>> jvpy.shape
        # torch.Size([6, 7, 3])
        # >>> y.shape
        # torch.Size([6, 7, 3])
        # >>> torch.allclose(y,f(x,z))
        # True
        # >>> jac_x,jac_z,_ = jacfwdb(f,x,z,batchdims=2)
        # >>> torch.allclose(jvpy,(jac_x*p[...,None,:]).sum(-1))
        # True
    """
    lenx = len(x) 
    assert len(x)>=1
    lenp = len(p)
    assert len(p)>=1
    batch_shape = list(x[0].shape[:batchdims])
    for i,xi in enumerate(x): assert list(xi.shape[:batchdims])==batch_shape, "x input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for i,pi in enumerate(p): assert list(pi.shape[:batchdims])==batch_shape, "p input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for k,v in batch_kwargs.items(): assert list(v.shape[:batchdims])==batch_shape, "batch_kwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_batch_kwargs = len(batch_kwargs) 
    batch_kwargs_keys = list(batch_kwargs.keys())
    def vjpfwrap(*inputs):
        x,p,batch_kwargs_vals = inputs[:lenx],inputs[lenx:(lenx+lenp)],inputs[(lenx+lenp):]
        y,vjpfwrap_inner = torch.func.vjp(lambda *x: f(*x,*batch_kwargs_vals),*x)
        vjpywrap = vjpfwrap_inner(p[0] if isinstance(y,torch.Tensor) else p)
        return y,vjpywrap
    vjpfwrapvec = torch.vmap(vjpfwrap,in_dims=(0,)*(lenx+lenp+len_batch_kwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=batchdims-1) if batchdims>0 else xi[None,...] for xi in x]
    p_input = [pi.flatten(end_dim=batchdims-1) if batchdims>0 else pi[None,...] for pi in p]
    batch_kwargs_vals_input = [batch_kwargs[key].flatten(end_dim=batchdims-1) if batchdims>0 else batch_kwargs[key][None,...] for key in batch_kwargs_keys]
    y,vjpys = vjpfwrapvec(*x_input,*p_input,*batch_kwargs_vals_input)
    if isinstance(y,torch.Tensor): y = (y,)
    if batchdims==0:
        return *(vjpy[0] for vjpy in vjpys),*(yk[0] for yk in y)
    else:
        return *(vjpy.reshape(batch_shape+list(vjpy.shape[1:])) for jacy in vjpys),*(yk.reshape(batch_shape+list(yk.shape[1:])) for yk in y)
