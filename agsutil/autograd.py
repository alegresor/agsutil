import torch 

def gradb(f, *x, batched_kwargs={}, batchdims=0, chunk_size=None):
    r"""
    Batched gradient and function evaluation 

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
        >>> grady_x,y = gradb(f,x,batched_kwargs={"z":z},batchdims=2)
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
    for k,v in batched_kwargs.items(): assert list(v.shape[:batchdims])==batch_shape, "batched_kwargs['%s'].shape = %s, but expected first dims to match batch_shape = %s"%(k,list(v.shape),list(batch_shape))
    len_batched_kwargs = len(batched_kwargs) 
    batched_kwargs_keys = list(batched_kwargs.keys())
    def fwrap(*inputs):
        x = inputs[:lenx]
        batched_kwargs_vals = inputs[lenx:(lenx+len_batched_kwargs)]
        batched_kwargs = {batched_kwargs_keys[l]:batched_kwargs_vals[l] for l in range(len_batched_kwargs)}
        y = f(*x,**batched_kwargs)
        return y,y
    gradfwrap = torch.func.grad(fwrap,argnums=tuple(i for i in range(len(x))),has_aux=True)
    xbshapes = [xi.shape[:batchdims] for xi in x]
    batched_kwargs_shapes = [(batched_kwargs[k].shape[:batchdims]) for k in batched_kwargs_keys]
    gradfwrapvec = torch.vmap(gradfwrap,in_dims=(0,)*(lenx+len_batched_kwargs),chunk_size=chunk_size)
    x_input = [xi.flatten(end_dim=batchdims-1) if batchdims>0 else xi[None,...] for xi in x]
    batched_kwargs_vals_input = [batched_kwargs[key].flatten(end_dim=batchdims-1) if batchdims>0 else batched_kwargs[key][None,...] for key in batched_kwargs_keys]
    gradys,y = gradfwrapvec(*x_input,*batched_kwargs_vals_input)
    if batchdims==0:
        return *(grady[0] for grady in gradys),y[0]
    else:
        return *(grady.reshape(batch_shape+list(grady.shape[1:])) for grady in gradys),y.reshape(batch_shape+list(y.shape[1:]))
