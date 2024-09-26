Examples
========

How to get a numpy object from the CNS monitor raw data using pycns ?

.. code-block:: python

    from pycns import CnsReader
    from pathlib import Path

    # set the path to the raw data folder
    raw_folder = '/XXX/YYY/ZZZ/'

    # make a Pathlib object
    raw_folder = Path(raw_folder)

    # read the data folder
    cns_reader = CnsReader(raw_folder)

    # show all available streams
    print(cns_reader)

    # show all available streams and their sampling rate
    print(cns_reader.streams)

    # use get_data() with isel method for an index based selection of data
    # to load a chunk of 'CO2' stream (for example) handled with numpy.datetime64
    sig, times = cns_reader.streams['CO2'].get_data(isel=slice(100_000, 110_000))

    # use get_data() with sel method for a timestamp based selection of data
    # to load a chunk of 'CO2' stream (for example) handled with numpy.datetime64
    start = '2021-01-08T00:10:15' # set a start date
    stop = '2021-01-08T00:30:50' # set a stop date
    sig, = cns_reader.streams['CO2'].get_data(sel=slice(start, stop))

    # use apply_gain = True to load data with their proper unit (default = False)
    start = '2021-01-08T00:10:15' # set a start date
    stop = '2021-01-08T00:30:50' # set a stop date
    sig = cns_reader.streams['CO2'].get_data(sel=slice(start, stop), apply_gain = True)

    # use with_times = True to return timestamps vector
    start = '2021-01-08T00:10:15' # set a start date
    stop = '2021-01-08T00:30:50' # set a stop date
    sig, times = cns_reader.streams['CO2'].get_data(sel=slice(start, stop), with_times = True, apply_gain = True)

How to load events from the Events.xml file ?

.. code-block:: python

    from pycns import CnsReader
    from pathlib import Path

    # load data using the same procedure than previously explained but using with_events = True and setting the timezone of the events
    raw_folder = '/XXX/YYY/ZZZ/'
    raw_folder = Path(raw_folder)
    cns_reader = CnsReader(raw_folder, with_events = True, event_time_zone='Europe/Paris') # setting timezone is mandatory.

    # show the events
    print(cns_reader.events) # can be handled by Pandas DataFrame using pd.DataFrame(cns_reader.events)

How to get a Xarray Dataset object from the CNS monitor raw data using pycns ?

.. code-block:: python

    from pycns import CnsReader
    from pathlib import Path

    # set the path to the raw data folder
    raw_folder = '/XXX/YYY/ZZZ/'

    # make a Pathlib object
    raw_folder = Path(raw_folder)

    # read the data folder
    cns_reader = CnsReader(raw_folder)

    # export some streams to Xarray Dataset, containing one DataArray per stream with various sampling rates.
    # Note that gain is automatically applied
    stream_names = ['ECG_II', 'RESP', 'EEG']
    start = '2021-01-08T00:10:15'
    stop = '2021-01-08T00:30:52'
    ds = cns_reader.export_to_xarray(stream_names, start=start, stop=stop)

    # export some streams to Xarray Dataset with a resample on common time base by using resample = True and setting a common sample rate
    # warning : this resampling method should ideally be used to upsample streams to the highest sample rate of the chosen streams. Aliasing may appear while down sampling.
    stream_names = ['ECG_II', 'RESP', 'EEG']
    start = '2021-01-08T00:10:15'
    stop = '2021-01-08T00:30:52'
    ds = cns_reader.export_to_xarray(stream_names, start=start, stop=stop, resample=True, sample_rate=256.)

How to use the toolbox as a viewer ? 
Note : should be used on a jupyter notebook using %matplotlib widgets (pip install -U ipywidgets==7.7.1)

.. code-block:: python

    from pycns import CnsReader, get_viewer
    from pathlib import Path

    # set the path to the raw data folder
    raw_folder = '/XXX/YYY/ZZZ/'

    # make a Pathlib object
    raw_folder = Path(raw_folder)

    # read the data folder
    cns_reader = CnsReader(raw_folder, with_events = True, event_time_zone = 'Europe/Paris') # events can be useful in the viewer to jump from event to event

    # easy viewer to navigate (this work only in jupyter)
    viewer = get_viewer(cns_reader)
    display(viewer)

    # select some streams
    viewer = get_viewer(cns_reader, stream_names=['CO2','ECG_II'])
    display(viewer)

    # set with_events = True to add a panel to jump from event to event
    viewer = get_viewer(cns_reader, stream_names=['CO2','ECG_II'], with_events = True)
    display(viewer)

How to customize the viewer ?

.. code-block:: python

    from pycns import CnsReader, get_viewer
    from pathlib import Path
    import scipy # for some examples of external views

    # set the path to the raw data folder
    raw_folder = '/XXX/YYY/ZZZ/'

    # make a Pathlib object
    raw_folder = Path(raw_folder)

    # read the data folder
    cns_reader = CnsReader(raw_folder, with_events = True, event_time_zone = 'Europe/Paris') # events can be useful in the viewer to jump from event to event

    # custom views can be add to the viewer. Such external views can be given to the get_viewer() function the ext_plots parameter with has to be fed with a dictionnary of python class.
    # Each of this class should return a figure with abscissa corresponding to times comprised between a start datetime (t0) and a stop datetime (t1).
    # Let's create an example of such a class which aims to display for example a spectrogram (Density Spectral Array) of one channel of an eeg stream. Such class could be prepared and imported from a dedicated python script.

    class Spectrogram_eeg:
    name = 'Spectrogram_eeg'

    def __init__(self, eeg_stream, chan_name, win_size_secs, lf=None, hf=None):
        self.eeg_stream = eeg_stream
        self.win_size_secs = win_size_secs # window size in seconds (welch method)
        self.chan_name = chan_name
        self.lf = lf
        self.hf = hf
        
    def plot(self, ax, t0, t1):
        eeg_stream = self.eeg_stream
        srate = eeg_stream.sample_rate # get srate
        chan_name = self.chan_name # get chan name
        chan_ind = eeg_stream.channel_names.index(chan_name) # get chan index
        sigs, datetimes = self.eeg_stream.get_data(sel=slice(t0, t1), with_times=True, apply_gain=True) # get data from all channels from t0 to t1 with proper units (gain)
        sig = sigs[:,chan_ind] # sel signal from the selected channel
 
        lf = self.lf # get low frequency cut
        hf = self.hf # get high frequency cut
        
        freqs, times_spectrum_s, Sxx = scipy.signal.spectrogram(sig, fs = srate, nperseg = int(self.win_size_secs * srate)) # compute spectrogram of the signal
        times_spectrum = (times_spectrum_s * 1e6) * np.timedelta64(1, 'us') + datetimes[0] # construct datetime vector of the spectrogram

        # prepare a frequency mask from the spectrogram as lf and hf parameters indicate
        if lf is None and hf is None:
            f_mask = (freqs>=freqs[0]) and (freqs<=freqs[-1])
        elif lf is None and not hf is None:
            f_mask = (freqs<=hf)
        elif not lf is None and hf is None:
            f_mask = (freqs>=lf)
        else:
            f_mask = (freqs>=lf) & (freqs<=hf)
        
        ax.pcolormesh(times_spectrum, freqs[f_mask], Sxx[f_mask,:]) # apply mask and plot
        ax.set_ylim(lf, hf)
        ax.set_ylabel(f'Spectro EEG\n{chan_name}\nFrequency (Hz)') # set an ylabel


    # prepare a dictionnary of external plots
    ext_plots = {
    'DSA':Spectrogram_eeg(eeg_stream = cns_reader.streams['EEG'], chan_name='C4', win_size_secs=2, lf=7, hf=13),
    }

    viewer = get_viewer(cns_reader, stream_names=['EEG'], ext_plots=ext_plots, with_events=True) # use ext_plots parameter
    display(viewer)

    