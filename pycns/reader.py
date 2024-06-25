from pathlib import Path
import xml
import xml.etree.ElementTree

# TODO rewrite time vector

import numpy as np
import xarray as xr
import scipy.interpolate
# this is needed for the 24bit trick
from numpy.lib.stride_tricks import as_strided

import dateutil

translation = {
    'ABP_Dias': 'DAP',
}


def explore_folder(folder, with_quality=False, with_processed=False, translate=False):

    name_streams = {}
    if with_processed:
        pattern = '**/*,data'
    else:
        pattern = '*,data'

    for filename in folder.glob(pattern):
        # print(filename.stem)
        # print(filename.name)
        fields = filename.name.split(',')
        f0 = fields[0]
        f1 = fields[1]
        f2 = fields[2]
        
        if f2 == 'Event':
            continue

        if not with_quality and 'Quality' in f2:
            continue
        
        if 'Processed' not in filename.parent.stem:
            if f1 =='na' and f2 =='SampleSeries':
                key = f0
            elif f1 =='na' and f2 =='Numeric':
                key = f0
            elif f1 !='na' and f2 =='Numeric':
                key = f0 + '_' + f1
            elif f1 == 'Composite' and f2 =='SampleSeries':
                key = f0
            elif f1 == 'Composite' and f2 !='SampleSeries':
                key = f0 + '_' + f2
            elif f1 !='na' and f2 =='SampleSeries':
                key = f0 + '_' + f1
            else:
                key = f0 + '_' + f1 + '_' + f2
        
            if translate and key in translation:
                key = translation[key]
        else:
            if 'PowerSpectrum' in filename.name:
                continue

            key = 'Processed_' + f0 + '_' + f1
            for field in fields[4:-1]:
                key = key + '_' + field

        assert key not in name_streams
        
        name_streams[key] = filename

    return name_streams


class CnsReader:
    """
    Class for exploring and reading a CNS folder.
    
    
    
    """
    def __init__(self, folder, with_quality=False, with_processed=False, translate=False, with_events=False,
                 event_time_zone=None,
                 ):
        self.folder = Path(folder)

        self.stream_names = explore_folder(folder, with_quality=with_quality, 
                                           with_processed=with_processed, translate=translate)

        self.streams = {}
        for name, raw_file in self.stream_names.items():
            # self.streams[name] = CnsStream(raw_file, name)
            try:
                self.streams[name] = CnsStream(raw_file, name)
            except:
                pass
                print('Problem to read this file:', raw_file)
        
        self.events = None
        event_file = self.folder /  'Events.xml'
        if with_events and event_file.is_file():
            assert event_time_zone is not None, "To read event you need to provide event_time_zone"
            self.events = read_events_xml(event_file, time_zone=event_time_zone)

    def __repr__(self):
        txt = f'CnsReader: {self.folder.stem}\n'
        txt += f'{len(self.stream_names)} streams : {list(self.stream_names.keys())}'
        if self.events is not None:
            n_events = self.events['name'].size
            txt += f'\nEvents: {n_events}'


        return txt
    
    def export_to_xarray(self, stream_names,
                         start=None, stop=None,
                         folder=None, resample=False, sample_rate=None):
        """
        Export to several streams a big xarray Dataset.
        Can be done in memory or in a zarr folder.
        Every stream can keep the original time vector or resample to a unique time vector.

        """

        # output
        if folder is None:
            # in memory Dataset
            ds = xr.Dataset()
        else:
            # zarr folder
            folder = Path(folder)
            assert not folder.exists(), f'{folder} already exists'

        # time range
        if start is None:
            start = np.min([self.streams[name].get_times()[0] for name in stream_names])
        if stop is None:
            stop = max([self.streams[name].get_times()[-1] for name in stream_names])
        start = np.datetime64(start, 'us')
        stop = np.datetime64(stop, 'us')



        if resample:
            assert sample_rate is not None
            period_ns = np.int64(1/sample_rate * 1e9)
            common_times = np.arange(start.astype('datetime64[ns]').astype('int64'),
                            stop.astype('datetime64[ns]').astype('int64'),
                            period_ns).astype('datetime64[ns]')

        for stream_name in stream_names:
            stream = self.streams[stream_name]
            sig, times = stream.get_data(sel=slice(start, stop), with_times=True, apply_gain=True)

            if not resample:
                # every array have its own time vector
                sig, times = stream.get_data(sel=slice(start, stop), with_times=True, apply_gain=True)
                time_dim = f'times_{stream_name}'
                dims = (time_dim, )
                coords = {time_dim: times}
            else:
                # add a few sample on border for better reample                
                delta = np.timedelta64(int(5 * 1e6 / stream.sample_rate), 'us')
                sig, times = stream.get_data(sel=slice(start - delta, stop + delta), with_times=True, apply_gain=True)

                # resample
                dims = ('times', )
                coords = {'times': common_times}
                times = times.astype('datetime64[ns]')
                f = scipy.interpolate.interp1d(times.astype('int64'), sig, kind='linear', axis=0,
                                           copy=True, bounds_error=False,
                                           fill_value=np.nan, assume_sorted=True)
                sig = f(common_times.astype('int64'))

            # channels when 2D
            if sig.ndim == 2 and stream.channel_names is not None:
                coords['channels'] = stream.channel_names
                dims = dims + ('channels', )
            
            arr = xr.DataArray(sig, dims=dims, coords=coords)
            if not resample:
                arr.attrs['sample_rate'] = stream.sample_rate
            else:
                arr.attrs['sample_rate'] = sample_rate
            
            if folder is None:
                ds[stream_name] = arr
            else:
                ds = xr.Dataset()
                ds[stream_name] = arr
                ds.to_zarr(folder, mode='a')

        if folder is not None:
            ds = xr.open_zarr(folder)

        return ds

        

dtype_index = [
    ('sample_ind', 'uint64'),
    ('datetime', 'datetime64[us]'),
    ('sample_interval_fract', 'uint32'),
    ('sample_interval_integer', 'uint32'),
    ('data_id', 'uint8'),
    ('cheksum', 'uint8'),
    ('bytes_per_sample', 'uint16'),
]


class CnsStream:
    def __init__(self, raw_file, name=None):
        raw_file = Path(raw_file)
        self.name = name
        
        name = raw_file.name
        self.raw_file = raw_file
        self.index_file = raw_file.parent / name.replace(',data', ',index')
        self.settings_file = raw_file.parent / name.replace(',data', ',settings')
        
        data_type = name.split(',')[3]
        
        # read time index
        self.index = np.memmap(self.index_file, mode='r', dtype=dtype_index)

        sample_interval_us = self.index['sample_interval_integer'] + (self.index['sample_interval_fract'] / 2 **32)
        self.sample_rate = np.mean(1e6 / sample_interval_us)


        # parse settings (units, gain, channel name)
        self.gain = None
        self.offset = None
        self.channel_names = None
        self.units = None

        with open(self.settings_file, encoding='iso-8859-15') as f:
            tree = xml.etree.ElementTree.parse(f)
        root = tree.getroot()
        units = root.find('Units')
        if units is not None:
            self.units = units.text
        if data_type == 'Integer':
            pass
        elif data_type == 'Composite':
            self.channel_names = []
            for e in root.find('CompositeElements'):
                chan_txt = e.attrib['type'].split(',')
                chan_name = chan_txt[1]
                self.channel_names.append(chan_name)
        
        if data_type in ('Integer', 'Composite'):
            conv_child = root.find('SampleConversion')
            if conv_child is not None:
                conv_txt = conv_child.text

                conv = [float(e) for e in conv_txt.split(',')]
                if conv[0] == -conv[1] and conv[2] == -conv[3]:
                    self.gain = conv[3] / conv[1]
                    self.offset = 0
                else:
                    raise NotImplementedError('Non symetric gain/offset scalling factor')


        self.need_24_bit_convert = False
        # read data buffer
        if data_type == 'Integer':
            self.raw_data = np.memmap(raw_file, mode='r', dtype='int32')
            self.shape = self.raw_data.shape
        elif data_type == 'Float':
            self.raw_data = np.memmap(raw_file, mode='r', dtype='float32')
            self.shape = self.raw_data.shape
        elif data_type == 'Composite':
            num_channels = len(self.channel_names)
            # check all packet have same dtype
            assert np.all(self.index['data_id'] == self.index['data_id'][0])
            
            if self.index['data_id'][0] - 0x80 == 0x05:
                # 24bits
                self.need_24_bit_convert = True
                
                
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
                #~ t0 = time.perf_counter()
                flat_data_uint24 = np.memmap(raw_file, mode='r', dtype='u1')
                num_frames = flat_data_uint24.size // (num_channels * 3)
                
                # this is needed because the last point will be outside
                
                #~ print('num_frames', num_frames, type(num_frames))
                #~ print('num_channels', num_channels, type(num_channels))
                # data_uint24 = flat_data_uint24.reshape(num_frames, num_channels, 3)

                # we need to remove the last sample to be sure that the last is not outside
                num_frames = num_frames - 1
                # self.raw_data = flat_data_uint24[:-(num_channels*3) + 1]
                new_size = (num_frames * num_channels * 3)
                new_size += 4 - new_size % 4
                self.raw_data = flat_data_uint24[:new_size]
                self.data_strided = as_strided(self.raw_data.view('uint32'),
                                               strides=(num_channels * 3, 3,),
                                               shape=(num_frames, num_channels))
                
                self.shape = (num_frames, num_channels)
                
                
                #~ data_uint32 = np.zeros((num_frames, num_channels, 4), dtype='u1')
                #~ data_uint32[:, :, :3] = data_uint24
                #~ data = data_uint32.flatten().view('u4').reshape(num_frames, num_channels)
                #~ t1 = time.perf_counter()
                #~ print(t1 - t0)


            elif self.index['data_id'][0] - 0x80 == 0x04:
                # 32bits
                data = np.memmap(raw_file, mode='r', dtype='int32')
                self.raw_data = data.reshape(-1, num_channels)
                self.shape = self.raw_data.shape
            else:
                raise NotImplementedError('Composite data type not known')
            
        else:
            raise ValueError
    
    def __repr__(self):
        if self.name is None:
            name = self.raw_file.name
        else:
            name = self.name
        txt = f'CnsStream {name}  rate:{self.sample_rate:0.0f}Hz  shape:{self.shape}'
        return txt
    
    def get_times(self):
        

        if hasattr(self, '_times'):
            # TODO make option to cache or not the times
            return self._times
    
        length = self.shape[0]
        times = np.zeros(length, dtype='datetime64[us]')
        
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
        
        index = self.index
        # strategy 2 : use the sample interval per block
        for i in range(index.size):
            ind0 = index[i]['sample_ind']
            datetime0 = index[i]['datetime']
            if i < (index.size - 1):
                ind1 = index[i +1]['sample_ind']
            else:
                ind1 = np.uint64(length)
            sample_interval_us = index[i]['sample_interval_integer'] + (index[i]['sample_interval_fract'] / 2 **32)
            local_times = datetime0 + (np.arange(ind1- ind0) * sample_interval_us).astype('timedelta64[us]')
            times[ind0:ind1] = local_times
        
        self._times = times
        return times


    def get_data(self, isel=None, sel=None, with_times=False, apply_gain=False):
        """

        isel: selection by integer range
        sel: selection by datetime slice
        
        """
        
        times = self.get_times()

        i0 = 0
        i1 = self.shape[0]
        

        if sel is not None:
            assert isel is None
            # TODO find somthing faster!!!!! only with the index
            
            if sel.start is not None:
                i0 = np.searchsorted(times, np.datetime64(sel.start))
            if sel.stop is not None:
                i1 = np.searchsorted(times, np.datetime64(sel.stop))
            
            assert sel.step is None 

        elif isel is not None:
            # print(isel.start)
            if isel.start is not None:
                i0 = int(isel.start)
            if isel.stop is not None:
                i1 = int(isel.stop)
            assert isel.step is None 

        # print(i0, i1)

        
        if self.need_24_bit_convert:
            data_32bit = self.data_strided[i0 : i1]
            # need to remove the first 8bits
            data = data_32bit & 0x00ffffff
            # handle the sign of 24 bits and shift it
            signs = (data > 0x0080ffff).astype('uint32')
            signs *= 0xff000000
            data = (data | signs).view('int32')
        else:
            data = self.raw_data[i0: i1]

        if apply_gain and self.gain is not None:
            data = data * self.gain
            if self.offset is not None and self.offset != 0.:
                data += self.offset

        if with_times:
            times = times[i0 :i1]
            return data, times
        else:
            return data


def convert_from_time_zone_to_gmt(start_time, time_zone):
    import pandas as pd
    return pd.Series(start_time).dt.tz_localize(time_zone).dt.tz_convert('GMT').values


def read_events_xml(event_file, time_zone):
    with open(event_file, encoding='utf-8') as f:
        tree = xml.etree.ElementTree.parse(f)
    root = tree.getroot()

    fields = {
        'Name': ('name', 'str'),
        'StartTime': ('start_time', 'datetime64[us]'),
        'Duration': ('duration', 'float64'),
        'Description': ('description', 'str'),
    }

    events = {}
    for field_name, (key, dtype) in fields.items():
        events[key] = []

    for e in root.findall('Event'):
        for field_name, (key, dtype) in fields.items():
            value = e.find(field_name).text
            if value is None:
                value = ''
            value = np.dtype(dtype).type(value)
            events[key].append(value)

    for field_name, (key, dtype) in fields.items():
        events[key] = np.array(events[key], dtype=dtype)
    
    events['start_time'] = convert_from_time_zone_to_gmt(events['start_time'], time_zone)

    return events
