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
        >>> grady,y = gradb(f,x)
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
        >>> grady_x,grady_z,y = gradb(f,x,z)
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
        >>> x = torch.rand(2,3,4,5,generator=rng) 
        >>> grady,y = gradb(f,x,batchdims=3)
        >>> grady.shape 
        torch.Size([2, 3, 4, 5])
        >>> y.shape 
        torch.Size([2, 3, 4])
        >>> torch.allclose(y,f(x))
        True
        >>> torch.allclose(grady,2*x)
        True

        >>> f = lambda x,z: (x**2*z**2).sum(-1)
        >>> x = torch.rand(2,3,4,5,generator=rng) 
        >>> z = torch.rand(2,3,4,5,generator=rng) 
        >>> grady_x,grady_z,y = gradb(f,x,z,batchdims=3)
        >>> grady_x.shape 
        torch.Size([2, 3, 4, 5])
        >>> grady_z.shape 
        torch.Size([2, 3, 4, 5])
        >>> y.shape
        torch.Size([2, 3, 4])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(grady_x,2*x*z**2)
        True
        >>> torch.allclose(grady_z,x**2*2*z)
        True
        
        >>> f = lambda x,z: (x**2*z**2).sum((-2,-1))
        >>> x = torch.rand(2,3,4,5,generator=rng) 
        >>> z = torch.rand(2,3,4,5,generator=rng) 
        >>> grady_x,y = gradb(f,x,batch_kwargs={"z":z},batchdims=2)
        >>> grady_x.shape 
        torch.Size([2, 3, 4, 5])
        >>> y.shape
        torch.Size([2, 3])
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
    batch_kwargs_shapes = [(batch_kwargs[k].shape[:batchdims]) for k in batch_kwargs_keys]
    gradfwrapvec = torch.vmap(gradfwrap,in_dims=(0,)*(lenx+len_batch_kwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=batchdims-1) if batchdims>0 else xi[None,...] for xi in x]
    batch_kwargs_vals_input = [batch_kwargs[key].flatten(end_dim=batchdims-1) if batchdims>0 else batch_kwargs[key][None,...] for key in batch_kwargs_keys]
    gradys,y = gradfwrapvec(*x_input,*batch_kwargs_vals_input)
    if batchdims==0:
        return *(grady[0] for grady in gradys),y[0]
    else:
        return *(grady.reshape(batch_shape+list(grady.shape[1:])) for grady in gradys),y.reshape(batch_shape+list(y.shape[1:]))

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
        >>> jacy,y = jacfwdb(f,x)
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
        >>> jacy_x,jacy_z,y = jacfwdb(f,x,z)
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
        >>> jacy,y = jacfwdb(f,x,batchdims=2)
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
        >>> jacy_x,jacy_z,y = jacfwdb(f,x,z,batchdims=2)
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
        >>> jacy_x,y = jacfwdb(f,x,batch_kwargs={"z":z},batchdims=2)
        >>> jacy_x.shape 
        torch.Size([3, 4, 2, 2, 5, 6])
        >>> y.shape
        torch.Size([3, 4, 2, 2])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None,None]*x[...,None,None,:,:]**torch.arange(1,3)[:,None,None,None]*z[...,None,None,:,:]**torch.arange(3,5)[None,:,None,None])
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
    jacfwrap = torch.func.jacfwd(fwrap,argnums=tuple(i for i in range(len(x))),has_aux=True)
    batch_kwargs_shapes = [(batch_kwargs[k].shape[:batchdims]) for k in batch_kwargs_keys]
    jacfwrapvec = torch.vmap(jacfwrap,in_dims=(0,)*(lenx+len_batch_kwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=batchdims-1) if batchdims>0 else xi[None,...] for xi in x]
    batch_kwargs_vals_input = [batch_kwargs[key].flatten(end_dim=batchdims-1) if batchdims>0 else batch_kwargs[key][None,...] for key in batch_kwargs_keys]
    jacys,y = jacfwrapvec(*x_input,*batch_kwargs_vals_input)
    if batchdims==0:
        return *(jacy[0] for jacy in jacys),y[0]
    else:
        return *(jacy.reshape(batch_shape+list(jacy.shape[1:])) for jacy in jacys),y.reshape(batch_shape+list(y.shape[1:]))

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
        >>> jacy,y = jacrevb(f,x)
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
        >>> jacy_x,jacy_z,y = jacrevb(f,x,z)
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
        >>> jacy,y = jacrevb(f,x,batchdims=2)
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
        >>> jacy_x,jacy_z,y = jacrevb(f,x,z,batchdims=2)
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
        >>> jacy_x,y = jacrevb(f,x,batch_kwargs={"z":z},batchdims=2)
        >>> jacy_x.shape 
        torch.Size([3, 4, 2, 2, 5, 6])
        >>> y.shape
        torch.Size([3, 4, 2, 2])
        >>> torch.allclose(y,f(x,z))
        True
        >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None,None]*x[...,None,None,:,:]**torch.arange(1,3)[:,None,None,None]*z[...,None,None,:,:]**torch.arange(3,5)[None,:,None,None])
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
    jacfwrap = torch.func.jacrev(fwrap,argnums=tuple(i for i in range(len(x))),has_aux=True)
    batch_kwargs_shapes = [(batch_kwargs[k].shape[:batchdims]) for k in batch_kwargs_keys]
    jacfwrapvec = torch.vmap(jacfwrap,in_dims=(0,)*(lenx+len_batch_kwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=batchdims-1) if batchdims>0 else xi[None,...] for xi in x]
    batch_kwargs_vals_input = [batch_kwargs[key].flatten(end_dim=batchdims-1) if batchdims>0 else batch_kwargs[key][None,...] for key in batch_kwargs_keys]
    jacys,y = jacfwrapvec(*x_input,*batch_kwargs_vals_input)
    if batchdims==0:
        return *(jacy[0] for jacy in jacys),y[0]
    else:
        return *(jacy.reshape(batch_shape+list(jacy.shape[1:])) for jacy in jacys),y.reshape(batch_shape+list(y.shape[1:]))

def jvpb(f, x, p, batchdims=0, chunk_size=None):
    r"""
    Batched `torch.func.jvp` with function evaluation (forward-mode auto-diff)

    Args:
        f (callable): Function to compute `torch.func.jvp` of.
        x (Tuple): (batched) `Torch.Tensor` inputs.
        p (Tuple): (batched) `Torch.Tensor` primals.
        batchdims (int): number of batch dimensions 
        chunk_size (int): to be passed into `torch.func.vmap`. 
    
    Returns: 
        jvps (tuple): `torch.Tensor` jacobian vector products, one with respect to each item in `x`. 
        y (torch.Tensor): Function evaluations. 
    
    Examples: 
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(7)

        >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        >>> x = torch.rand(5,generator=rng)
        >>> p = torch.rand(5,generator=rng)
        >>> jacy,y = jvpb(f,(x,),(p,))
        >>> jacy.shape 
        torch.Size([3])
        >>> y.shape 
        torch.Size([3])
        >>> torch.allclose(y,f(x))
        True
        >>> torch.allclose(jacy,jacfwdb(f,x)[0]@p)
        True

        # >>> f = lambda x,z: (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum(-3)
        # >>> x = torch.rand(5,generator=rng) 
        # >>> z = torch.rand(5,generator=rng) 
        # >>> jacy_x,jacy_z,y = jacrevb(f,x,z)
        # >>> jacy_x.shape 
        # torch.Size([2, 2, 5])
        # >>> jacy_z.shape 
        # torch.Size([2, 2, 5])
        # >>> y.shape
        # torch.Size([2, 2])
        # >>> torch.allclose(y,f(x,z))
        # True
        # >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None]*x**torch.arange(1,3)[:,None,None]*z**torch.arange(3,5)[None,:,None])
        # True
        # >>> torch.allclose(jacy_z,x**torch.arange(2,4)[:,None,None]*torch.arange(3,5)[None,:,None]*z**torch.arange(2,4)[None,:,None])
        # True
        
        # >>> f = lambda x: (x[...,None]**torch.arange(2,5)).sum(-2)
        # >>> x = torch.rand(3,4,5,generator=rng) 
        # >>> jacy,y = jacrevb(f,x,batchdims=2)
        # >>> jacy.shape 
        # torch.Size([3, 4, 3, 5])
        # >>> y.shape 
        # torch.Size([3, 4, 3])
        # >>> torch.allclose(y,f(x))
        # True
        # >>> torch.allclose(jacy,torch.arange(2,5)[:,None]*x[...,None,:]**torch.arange(1,4)[:,None])
        # True

        # >>> f = lambda x,z: (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum(-3)
        # >>> x = torch.rand(3,4,5,generator=rng) 
        # >>> z = torch.rand(3,4,5,generator=rng) 
        # >>> jacy_x,jacy_z,y = jacrevb(f,x,z,batchdims=2)
        # >>> jacy_x.shape 
        # torch.Size([3, 4, 2, 2, 5])
        # >>> jacy_z.shape 
        # torch.Size([3, 4, 2, 2, 5])
        # >>> y.shape
        # torch.Size([3, 4, 2, 2])
        # >>> torch.allclose(y,f(x,z))
        # True
        # >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None]*x[...,None,None,:]**torch.arange(1,3)[:,None,None]*z[...,None,None,:]**torch.arange(3,5)[None,:,None])
        # True
        # >>> torch.allclose(jacy_z,x[...,None,None,:]**torch.arange(2,4)[:,None,None]*torch.arange(3,5)[None,:,None]*z[...,None,None,:]**torch.arange(2,4)[None,:,None])
        # True
        
        # >>> f = lambda x,z: (x[...,None,None]**torch.arange(2,4)[:,None]*z[...,None,None]**torch.arange(3,5)[None,:]).sum((-4,-3))
        # >>> x = torch.rand(3,4,5,6,generator=rng) 
        # >>> z = torch.rand(3,4,5,6,generator=rng) 
        # >>> jacy_x,y = jacrevb(f,x,batch_kwargs={"z":z},batchdims=2)
        # >>> jacy_x.shape 
        # torch.Size([3, 4, 2, 2, 5, 6])
        # >>> y.shape
        # torch.Size([3, 4, 2, 2])
        # >>> torch.allclose(y,f(x,z))
        # True
        # >>> torch.allclose(jacy_x,torch.arange(2,4)[:,None,None,None]*x[...,None,None,:,:]**torch.arange(1,3)[:,None,None,None]*z[...,None,None,:,:]**torch.arange(3,5)[None,:,None,None])
        # True
    """
    lenx = len(x) 
    assert len(x)>=1
    assert len(p)==lenx
    batch_shape = list(x[0].shape[:batchdims])
    for i,xi in enumerate(x): assert list(xi.shape[:batchdims])==batch_shape, "x input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    for i,pi in enumerate(p): assert list(pi.shape[:batchdims])==batch_shape, "p input %d has shape = %s, but expected first dims to match batch_shape = %s"%(i,list(xi.shape),list(batch_shape))
    def jvpfwrap(*xp):
        lenxp = len(xp)
        x,p = xp[:(lenxp//2)],xp[(lenxp//2):]
        return torch.func.jvp(f,x,p)
    jvpfwrapvec = torch.vmap(jvpfwrap,in_dims=(0,)*2*lenx,chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=batchdims-1) if batchdims>0 else xi[None,...] for xi in x]
    p_input = [pi.flatten(end_dim=batchdims-1) if batchdims>0 else pi[None,...] for pi in p]
    y,jvpys = jvpfwrapvec(*x_input,*p_input)
    if batchdims==0:
        return jvpys[0],y[0]
    else:
        return *(jvpy.reshape(batch_shape+list(jvpy.shape[1:])) for jvpy in jvpys),y.reshape(batch_shape+list(y.shape[1:]))
