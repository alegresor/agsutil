import torch 
import time

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
