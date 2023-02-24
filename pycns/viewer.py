# https://ipywidgets.readthedocs.io/en/7.6.3/examples/Widget%20Styling.html


import numpy as np

import xarray as xr

import matplotlib.pyplot as plt

import ipywidgets.widgets as W

import time

# TODO expandable layout for fiigure.canvas
# TODO make an absctract class to retrieve data from dataset or cnsreader



def make_timeslider(data, stream_names, time_range=None, width_cm=10):
    
    if isinstance(data, xr.Dataset):
        # TODO take in account the stream_names
        ds = data
        t_start = min([ds[name].values[0] for name in ds.coords if name.startswith('times')])
        t_stop = max([ds[name].values[-1] for name in ds.coords if name.startswith('times')])
    else:
        t_start = min([data.streams[name].get_times()[0].astype('datetime64[ns]') for name in stream_names])
        t_stop = max([data.streams[name].get_times()[-1].astype('datetime64[ns]') for name in stream_names])

    if time_range is None:
        time_range = [t_start, t_start + np.timedelta64(1, 'm')]
    

    time_label = W.Label(value=f'{time_range[0]}',
                         layout=W.Layout(width=f'6cm'))

    time_slider = W.IntSlider(
        orientation='horizontal',
        description='time:',
        value=np.int64(time_range[0]),
        min=np.int64(t_start),
        max=np.int64(t_stop),
        readout=False,
        continuous_update=False,
        #~ continuous_update=True,
        layout=W.Layout(width=f'{width_cm}cm')
    )

    delta = np.int64(np.diff(time_range)) / 1e9
    window_sizer = W.BoundedFloatText(value=delta, step=0.1, min=0.005,max=3600,
                                      description='win (s)',
                                     layout=W.Layout(width=f'4cm'))


    main_widget = W.HBox([time_slider, time_label, window_sizer])
    some_widgets = {"time_label": time_label, "time_slider": time_slider, "window_sizer" : window_sizer}

    # return widget, controller
    return main_widget, some_widgets


def make_channel_selector(data, stream_names, width_cm=10, height_cm=5):
    channels =[]
    if isinstance(data, xr.Dataset):
        ds = data
        for stream_name in ds.keys():
            if stream_names is not None and stream_name not in stream_names:
                continue
            arr = ds[stream_name]
            if arr.ndim == 1:
                channels.append(stream_name)
            elif ds[stream_name].ndim == 2:
                chan_coords = [k for k in arr.coords.keys() if not k.startswith('times')][0]
                chans = list(arr.coords[chan_coords].values)
                channels.extend([f'{stream_name}/{chan}' for chan in chans])
    else:
        for stream_name, stream in data.streams.items():
            if stream_names is not None and stream_name not in stream_names:
                continue
            if stream.channel_names is None:
                channels.append(stream_name)
            else:
                channels.extend([f'{stream_name}/{chan}' for chan in stream.channel_names])
    
    #~ chan_selected = [chan for chan in channels if '/' not in chan]
    chan_selected = channels[:2]
    
    channel_selector = W.SelectMultiple(
        options=channels,
        value=chan_selected,
        disabled=False,
        layout=W.Layout(width=f'{width_cm}cm', height=f'{height_cm}cm')
    )
    
    
    some_widgets = {'channel_selector': channel_selector}
    return channel_selector, some_widgets


class PlotUpdater:
    def __init__(self, data, widgets, fig):
        self.data = data
        self.widgets = widgets
        self.fig = fig

        self._is_dataset = isinstance(data, xr.Dataset)

        self.axs = None
        
        self.update_channel_visibility()

    # def __call__(self, chang):
    #     print(change)

    def update_time(self, change=None):
        if self.axs is None:
            self.update_channel_visibility(None)
        
        t0 = self.widgets['time_slider'].value
        t1 = t0 + int(self.widgets['window_sizer'].value * 1e9)
        
        print()
        print(t0, t1)
        t0 = np.datetime64(t0, 'ns')
        t1 = np.datetime64(t1, 'ns')
        if  not self._is_dataset:
            t0 = np.datetime64(t0, 'us')
            t1 = np.datetime64(t1, 'us')
        print(t0, t1)
        
        self.widgets['time_label'].value = f'{t0}'
        
        
        

        channels = self.get_visible_channels()
        
        previous_stream = None
        for i, channel in enumerate(channels):
            # TODO change the channel concept
            # this approach is not optimal because several call to the same stream when EEG
            
            ax = self.axs[i]
            
            #ax.clear()
            for l in ax.lines:
                # clear remove also labels
                l.remove()
            
            stream_name, chan = channels[i]
            
            if self._is_dataset:
                arr = self.data[stream_name]
                # the times is the first coords always
                time_coords = arr.dims[0]
                # time slice
                # time_coords = f'times_{stream_name}'
                d = {time_coords: slice(t0, t1)}
                arr = arr.sel(**d)
                if chan is not None:
                    #EEG channel slice
                    chan_coords = [k for k in arr.coords.keys() if not k.startswith('times_')][0]
                    d = {chan_coords : chan}
                    arr = arr.sel(**d)
                times = arr.coords[time_coords].values
                arr = arr.values
            else:
                #~ print(channel)
                stream = self.data.streams[stream_name]
                #~ tt0 = time.perf_counter()
                arr, times = stream.get_data(sel=slice(t0, t1), with_times=True, apply_gain=True)
                #~ tt1 = time.perf_counter()
                #~ print(stream, tt1 - tt0, arr.shape)
                
                if chan is not None:
                    chan_ind = list(stream.channel_names).index(chan)
                    arr = arr[:, chan_ind]
                    #~ print('ici', chan_ind, arr.shape)
                

            ax.plot(times, arr, color='k')
        
        # set scale on last axis
        ax = self.axs[-1]
        ax.set_xlim(t0, t1)

        self.fig.canvas.draw()
        #~ self.fig.canvas.flush_events()

    
    def update_channel_visibility(self, change=None):
        
        channels = self.get_visible_channels()
        self.fig.clear()
        n = len(channels)
        gs = self.fig.add_gridspec(nrows=n, ncols=1,
                                   left=0.15, right=.95, top=1., bottom=0.1,
                                   hspace=0)
        
        # self.axs = [self.fig.add_subplot(gs[i]) for i in range(n)]
        self.axs = []
        for i in range(n):
            stream_name, chan = channels[i]
            ax = self.fig.add_subplot(gs[i])
            if chan is None:
                label = stream_name
            else:
                label = chan
            if self._is_dataset:
                if 'units' in self.data[stream_name].attrs:
                    units = self.data[stream_name].attrs['units']
                    label = label + f' [{units}]'
            else:
                units = self.data.streams[stream_name].units
                if units is not None:
                    label = label + f' [{units}]'

            ax.set_ylabel(label)
            self.axs.append(ax)
        
        for ax in self.axs[:-1]:
            ax.sharex(self.axs[-1])
            ax.tick_params(labelbottom=False)
        
        # self.update_time()
    
    def full_refresh(self,  change=None):
        self.update_channel_visibility()
        self.update_time()
        
    def get_visible_channels(self):
        channels = []
        for k in self.widgets['channel_selector'].value:
            if '/' in k:
                stream_name, chan = k.split('/')
            else:
                stream_name, chan = k, None
            channels.append([stream_name, chan])
        return channels

cm = 1 / 2.54

def get_viewer(data, stream_names=None, width_cm=10, height_cm=8):

    ratios = [0.1, 0.8, 0.2]
    height_cm = width_cm * ratios[1]

    all_widgets = {}

    # figure
    with plt.ioff():
        output = W.Output()
        with output:
            # fig, ax = plt.subplots(figsize=(width_cm * cm, height_cm * cm))
            fig = plt.figure()
            fig.canvas.toolbar_visible = False

    # make fig expendable
    # print(fig.canvas.layout)
    
#             plt.show()
    # fig = plt.figure()
    
    if stream_names is None:
        if isinstance(data, xr.Dataset):
            ds = data
            stream_names = list(ds.keys())
        else:
            stream_names = list( data.streams.keys())



    # time slider
    time_slider, some_widgets = make_timeslider(data, stream_names, width_cm=width_cm, )
    all_widgets.update(some_widgets)

    # channels
    channel_selector, some_widgets = make_channel_selector(data, stream_names)
    all_widgets.update(some_widgets)


    updater = PlotUpdater(data, all_widgets, fig)
    all_widgets['time_slider'].observe(updater.update_time)
    all_widgets['window_sizer'].observe(updater.update_time)
    all_widgets['channel_selector'].observe(updater.full_refresh)
    
    

    
    tab0 = W.VBox([fig.canvas, time_slider])
    tab1 = W.VBox([channel_selector])
    
    main_widget = W.Tab(children=[tab0, tab1])
    main_widget.set_title(0, 'main')
    main_widget.set_title(1, 'options')
    

    # main_widget = W.AppLayout(
    #             center=fig.canvas,
    #             footer=time_slider,
    #             left_sidebar=None,
    #             right_sidebar=channel_selector,
    #             pane_heights=[0, 6, 1],
    #             pane_widths=ratios
    #         )

    main_widget._all_widgets = all_widgets
    return main_widget






