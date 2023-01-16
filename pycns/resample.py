import xarray as xr
import numpy as np
import scipy.interpolate
import pandas as pd

def resample(source_folder, target_folder, sample_rate=200., stream_names=None, ):
    """
    Resample a dataset
    """
    print(source_folder)
    print(target_folder)
    print(stream_names)
    
    source_ds = xr.open_zarr(source_folder)
    
    t_start, t_stop = None, None
    for stream_name in stream_names:
        times =source_ds[stream_name].coords[f'times_{stream_name}']
        if t_start is None:
            t_start = times.values[0]
            t_stop = times.values[-1]
        else:
            t_start = min(t_start, times.values[0])
            t_stop = max(t_stop, times.values[-1])
    print(t_start, t_stop)
    
    period_ns = int(1/sample_rate * 1e9)
    times = np.arange(int(t_start), int(t_stop)+1, period_ns).astype('datetime64[ns]')
    #times = pd.date_range(t_start, t_stop, freq=f'{period_ms}ms')
    print(times)
    print(times.shape)
    
    # exit()
    for stream_name in stream_names:
        
        source_arr = source_ds[stream_name]
        
        
        target_dims = ('time', )
        target_coords = {'times' : times}
        
        if source_arr.ndim >1:
            for dim in source_arr.dims[1:]:
                target_dims = target_dims + (dim, )
                target_coords[dim] = source_arr.coords[dim].values
        
        print(stream_name, target_dims, target_coords)
        
        
        target_arr = source_arr.interp(coords={f'times_{stream_name}': times}, method='linear')
        print(target_arr)

        target_arr = target_arr.rename({f'times_{stream_name}': 'times'})
        print(target_arr)
        
        print(source_arr.shape, target_arr.shape)
        
        target_ds = xr.Dataset()
        target_ds[stream_name] = target_arr
        target_ds.to_zarr(target_folder, mode='a')
                                       
        
        
#         target_coords = arr.coords
#         print(target_coords)
        
#         target_ds = xr.Dataset(coords=target_coords)
        
#         data_interpolated
        
#         target_arr = xr.DataArray(data_interpolated, dims=target_dims, coords=target_coords)
            
#         if 'units' in source_arr.attrs:
#             target_arr.attr['units'] = source_arr.attr['units']
#         target_ds[stream_name] = target_arr
        
#         if 'units' in settings:
#             arr.attrs['units'] = settings['units']
#         print('yep 5')
#         ds.to_zarr(zarr_folder, mode='a')
