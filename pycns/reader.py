from pathlib import Path
import xml
import xml.etree.ElementTree





class CnsReader:
    """
    Class for exploring and reading a CNS folder.
    
    
    
    """
    def __init__(self, folder):
        self.folder = Path(folder)
        

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
    def __init__(self, raw_file):
        raw_file = raw_filePath(raw_file)
        
        name = raw_file.stem
        self.raw_file =raw_file
        self.index_file = raw_file.parent / name.replace(',data', ',index')
        self.settings_file = raw_file.parent / name.replace(',data', ',settings')
        
        data_type = name.split(',')[3]
        
        # read time index
        self.time_index = np.memmap(index_file, mode='r', dtype=dtype_index)

        # parse settings (units, gain, channel name)
        self.gain = None
        self.offset = None
        tree = xml.etree.ElementTree.parse(settings_file)
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


        # read data buffer
        if data_type == 'Integer':
            self.raw_data = np.memmap(raw_file, mode='r', dtype='int32')
        elif data_type == 'Float':
            self.raw_data = np.memmap(raw_file, mode='r', dtype='float32')
        elif data_type == 'Composite':
            num_channels = len(self.channel_names)
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
                #~ t0 = time.perf_counter()
                flat_data_uint24 = np.memmap(raw_file, mode='r', dtype='u1')
                num_frames = flat_data_uint24.size // (num_channels * 3)
                data_uint24 = flat_data_uint24.reshape(num_frames, num_channels, 3)
                self.raw_data = data_uint24
                #~ data_uint32 = np.zeros((num_frames, num_channels, 4), dtype='u1')
                #~ data_uint32[:, :, :3] = data_uint24
                #~ data = data_uint32.flatten().view('u4').reshape(num_frames, num_channels)
                #~ t1 = time.perf_counter()
                #~ print(t1 - t0)


            elif index['data_id'][0] - 0x80 == 0x04:
                # 32bits
                data = np.memmap(raw_file, mode='r', dtype='int32')
                self.raw_data = data.reshape(-1, num_channels)
            else:
                raise NotImplementedError('Composite data type not known')
            
        else:
            raise ValueErrror
    
    



