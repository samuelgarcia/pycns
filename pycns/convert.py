"""

TODO:
  * alert
  * EEG
  * multiple name
  * MarkEvent

"""
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.io
import xarray as xr
import dask
import time

import xml
import xml.etree.ElementTree

from pathlib import Path
import shutil

import tqdm



dtype_index = [
    ('sample_ind', 'uint64'),
    ('datetime', 'datetime64[us]'),
    ('sample_interval_fract', 'uint32'),
    ('sample_interval_integer', 'uint32'),
    ('data_id', 'uint8'),
    ('cheksum', 'uint8'),
    ('bytes_per_sample', 'uint16'),
]


#

def read_one_stream(raw_file):
    name = raw_file.stem
    index_file = raw_file.parent / name.replace(',data', ',index')
    settings_file = raw_file.parent / name.replace(',data', ',settings')
    
    data_type = name.split(',')[3]
    
    # read index
    index = np.memmap(index_file, mode='r', dtype=dtype_index)
    
    #~ print(index['bytes_per_sample'])
    #~ exit()

    # read settings
    settings = {}
    tree = xml.etree.ElementTree.parse(settings_file)
    root = tree.getroot()
    settings['units'] = root.find('Units').text
    if data_type == 'Integer':
        conv_txt = root.find('SampleConversion').text
        conv = [float(e) for e in conv_txt.split(',')]
        if conv[0] == -conv[1] and conv[2] == -conv[3]:
            settings['gain'] = conv[1] / conv[3]
            settings['offset'] = 0
        else:
            raise NotImplementedErro('Non symetric gain/offset scalling factor')

    if data_type == 'Composite':
        settings['channel_names'] = []
        for e in root.find('CompositeElements'):
            chan_txt = e.attrib['type'].split(',')
            chan_name = chan_txt[1]
            settings['channel_names'].append(chan_name)
        
            
        #~ print(index['data_id'])
    #~ print(settings)
    #~ exit()

    
    # read data buffer
    
    if data_type == 'Integer':
        data = np.memmap(raw_file, mode='r', dtype='int32')
    elif data_type == 'Float':
        data = np.memmap(raw_file, mode='r', dtype='float32')
    elif data_type == 'Composite':
        
        
        num_channels = len(settings['channel_names'])
        # check all packet have same dtype
        assert np.all(index['data_id'] == index['data_id'][0])
        
        if index['data_id'][0] - 0x80 == 0x05:
            # 24bits
            
            
            # using strides trick
            # https://stackoverflow.com/questions/12080279/how-do-i-create-a-numpy-dtype-that-includes-24-bit-integers
            # t0 = time.perf_counter()
            # raw_data = np.memmap(raw_file, mode='r', dtype='u1')
            # num_frames = raw_data.size // (num_channels * 3)
            # data_int32 = as_strided(raw_data.view('int32'), strides=(num_channels * 3, 3,), shape=(num_frames, num_channels))
            #~ data_int32 = as_strided(raw_data.view('>u4'), strides=(num_channels * 3, 3,), shape=(num_frames, num_channels))
            # print(data_int32.shape)
            # data = data_int32 & 0x00ffffff
            #~ data = data_int32 & 0xffffff00
            # print(data.shape)
            # print(data[:10, 0])
            # t1 = time.perf_counter()
            # print(t1 - t0)

            # using naive copy
            t0 = time.perf_counter()
            flat_data_uint24 = np.memmap(raw_file, mode='r', dtype='u1')
            num_frames = flat_data_uint24.size // (num_channels * 3)
            data_uint24 = flat_data_uint24.reshape(num_frames, num_channels, 3)
            data_uint32 = np.zeros((num_frames, num_channels, 4), dtype='u1')
            data_uint32[:, :, :3] = data_uint24
            data = data_uint32.flatten().view('u4').reshape(num_frames, num_channels)
            t1 = time.perf_counter()
            print(t1 - t0)


        elif index['data_id'][0] - 0x80 == 0x04:
            # 32bits
            data = np.memmap(raw_file, mode='r', dtype='int32')
            data = data.reshape(-1, num_channels)
        else:
            raise NotImplementedError('Composite data type not known')
        
        
    else:
        raise ValueErrror
    
    #~ exit()
    #~ print(data.shape)
    
    
    #~ print(index)
    
        
    
    return data, index, settings


def make_time_vector(data, index):
    times = np.zeros(data.shape[0], dtype='datetime64[us]')
    
    # strategy 1 : interpolate between runs
    #~ for i in range(index.size):
        #~ ind0 = index[i]['sample_ind']
        #~ datetime0 = index[i]['datetime']
        #~ if i < (index.size - 1):
            #~ ind1 = index[i +1]['sample_ind']
            #~ datetime1 = index[i +1]['datetime']
        #~ else:
            #~ ind1 = np.uint64(data.shape[0])
            #~ sample_interval_us = index[i]['sample_interval_integer'] + (index[i]['sample_interval_fract'] / 2 **32)
            #~ datetime1 = datetime0 + int((ind1-ind0) * sample_interval_us)
        #~ local_times = np.linspace(datetime0.astype(int), datetime1.astype(int), ind1 - ind0, endpoint=False).astype("datetime64[us]")
        #~ times[ind0:ind1] = local_times
    
    # strategy 2 : use the sample interval per block
    for i in range(index.size):
        ind0 = index[i]['sample_ind']
        datetime0 = index[i]['datetime']
        if i < (index.size - 1):
            ind1 = index[i +1]['sample_ind']
        else:
            ind1 = np.uint64(data.shape[0])
        sample_interval_us = index[i]['sample_interval_integer'] + (index[i]['sample_interval_fract'] / 2 **32)
        local_times = datetime0 + (np.arange(ind1- ind0) * sample_interval_us).astype('timedelta64[us]')
        times[ind0:ind1] = local_times
    
    return times
    
def test_read_one_stream():
    folder = Path('/crnldata/projets_communs/PhysioNeuroRea/raw_data/')
    #~ raw_file = folder / 'SpO2,na,Numeric,Float,IntelliVue,data'
    #~ raw_file = folder / 'RESP,na,SampleSeries,Integer,IntelliVue,data'
    raw_file = folder / 'EEG,Composite,SampleSeries,Composite,Amp1020,data'
    
    
    data, index, settings = read_one_stream(raw_file)
    #~ print(settings)
    
    sample_interval_us = index['sample_interval_integer'] + (index['sample_interval_fract'] / 2 **32)
    #~ print(sample_interval_us)
    #~ print(1. / sample_interval_us)
    
    times = make_time_vector(data, index)
    print(times)
    
    import matplotlib.pyplot as plt
    
    
    fig, ax = plt.subplots()
    ax.plot(data[:100000, 0])
    
    fig, ax = plt.subplots()
    ax.scatter(index['sample_ind'], index['datetime'])
    
    #~ fig, ax = plt.subplots()
    #~ ax.scatter(index['sample_ind'], index['datetime'])
    #~ ax.plot(np.arange(data.shape[0]), times, color='g')
    
    plt.show()


def explore_folder(raw_folder):

    streams = {}
    for filename in raw_folder.glob('*,data'):
        # print()
        #~ print(filename.stem)
        fields = filename.stem.split(',')
        stream_name = fields[0]
        stream_type = fields[2]
        streams[(stream_name, stream_type)] = filename

    return streams

def test_explore_folder():
    folder = Path('/crnldata/projets_communs/PhysioNeuroRea/raw_data/P1')
    streams = explore_folder(folder)
    print(streams)
    

def convert_folder_to_dataset(raw_folder, zarr_folder, load_streams=None, progress_bar=True):
    assert not zarr_folder.exists(), 'out_zarr_folder already exists'
    
    
    streams = explore_folder(raw_folder)
    print(streams)

    #~ if load_streams is None:
        #~ load_streams = list(streams.keys())
        #~ print(load_streams)

    print(load_streams)
    
    if progress_bar:
        load_streams = tqdm.tqdm(load_streams)
    
    for stream_name, stream_type in streams:
        if load_streams is not None:
            if stream_name not in load_streams:
                print(stream_name, stream_name not in stream_name)
                continue
        
        
        if stream_type != 'SampleSeries':
            continue
        
        print()
        print(stream_name)
        
        raw_file = streams[(stream_name, stream_type)]
        print('yep 1')
        data, index, settings = read_one_stream(raw_file)
        #~ print(index)
        print('yep 2')
        
        if 'gain' in settings:
            data = data.astype('float32')
            data *= settings['gain']
            if settings['offset'] != 0:
                data += settings['offset']
        print(data.dtype)
        sample_interval_us = index['sample_interval_integer'] + (index['sample_interval_fract'] / 2 **32)
        
        sample_interval = np.mean(sample_interval_us) / 1e6
        
        times = make_time_vector(data, index)
        print(times.dtype)
        print('yep 3')
        #~ print(times)
        time_dim = f'times_{stream_name}'
        coords = {time_dim: times}
        dims = (time_dim, )
        if data.ndim == 2:
            coords['channels'] = settings['channel_names']
            dims = dims + ('channels', )
        
        ds = xr.Dataset(coords=coords)
        print('yep 4')
        ds[stream_name] = xr.DataArray(data, dims=dims, coords=coords)
        if 'units' in settings:
            ds[stream_name].attrs['units'] = settings['units']
        print('yep 5')
        ds.to_zarr(zarr_folder, mode='a')
        print('yep 6')
    



if __name__ == '__main__':
    # test_read_one_stream()
    test_explore_folder()