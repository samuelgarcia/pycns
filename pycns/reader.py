from pathlib import Path
import xml
import xml.etree.ElementTree

import numpy as np
# this is needed for the 24bit trick
from numpy.lib.stride_tricks import as_strided

translation = {
    'ABP_Dias': 'DAP',
}

def explore_folder(folder, with_quality=False, translate=False):

    name_streams = {}
    for filename in folder.glob('*,data'):
        # print()
        #~ print(filename.stem)
        fields = filename.stem.split(',')
        f0 = fields[0]
        f1 = fields[1]
        f2 = fields[2]
        
        if f2 == 'Event':
            continue
        
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
        if not with_quality and 'Quality' in f2:
            continue
        
        
        if translate and key in translation:
            key = translation[key]
        
        assert key not in name_streams
        
        name_streams[key] = filename

    return name_streams


class CnsReader:
    """
    Class for exploring and reading a CNS folder.
    
    
    
    """
    def __init__(self, folder, with_quality=False, translate=False):
        self.folder = Path(folder)

        self.stream_names = explore_folder(folder, with_quality=with_quality, translate=translate)

        self.streams = {}
        for name, raw_file in self.stream_names.items():
            self.streams[name] = CnsStream(raw_file, name)

    def __repr__(self):
        txt = f'CnsReader: {self.folder.stem}\n'
        txt +=f'{len(self.stream_names)} streams : {list(self.stream_names.keys())}'

        return txt

        

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
        
        name = raw_file.stem
        self.raw_file =raw_file
        self.index_file = raw_file.parent / name.replace(',data', ',index')
        self.settings_file = raw_file.parent / name.replace(',data', ',settings')
        
        data_type = name.split(',')[3]
        
        # read time index
        self.index = np.memmap(self.index_file, mode='r', dtype=dtype_index)

        # parse settings (units, gain, channel name)
        self.gain = None
        self.offset = None
        self.channel_names = None
        self.units = None

        tree = xml.etree.ElementTree.parse(self.settings_file)
        root = tree.getroot()
        self.units = root.find('Units').text
        if data_type == 'Integer':
            conv_txt = root.find('SampleConversion').text
            conv = [float(e) for e in conv_txt.split(',')]
            if conv[0] == -conv[1] and conv[2] == -conv[3]:
                self.gain = conv[1] / conv[3]
                self.offset = 0
            else:
                raise NotImplementedErro('Non symetric gain/offset scalling factor')
        elif data_type == 'Composite':
            self.channel_names = []
            for e in root.find('CompositeElements'):
                chan_txt = e.attrib['type'].split(',')
                chan_name = chan_txt[1]
                self.channel_names.append(chan_name)

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
                #~ print('num_frames', num_frames, type(num_frames))
                #~ print('num_channels', num_channels, type(num_channels))
                # data_uint24 = flat_data_uint24.reshape(num_frames, num_channels, 3)
                self.raw_data = flat_data_uint24
                self.data_strided = as_strided(self.raw_data.view('int32'), strides=(num_channels * 3, 3,), shape=(num_frames, num_channels))
                
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
            raise ValueErrror
    
    def __repr__(self):
        if self.name is None:
            name = self.raw_file.stem
        else:
            name = self.name
        txt = f'CnsStream: {name}'
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


    def get_data(self, isel=None, sel=None, with_times=False):
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
        else:
            data = self.raw_data[i0: i1]

        if with_times:
            times = times[i0 :i1]
            return data, times
        else:
            return data

            

            





