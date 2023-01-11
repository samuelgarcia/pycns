# https://ipywidgets.readthedocs.io/en/7.6.3/examples/Widget%20Styling.html


import numpy as np

import matplotlib.pyplot as plt

import ipywidgets.widgets as W

def make_timeslider(ds, time_range=None, width_cm=10):
    t_start = min([ds[name].values[0] for name in ds.coords if name.startswith('times_')])
    t_stop = max([ds[name].values[-1] for name in ds.coords if name.startswith('times_')])

    print(t_start, t_stop)
    print(int(t_start), int(t_stop))

    if time_range is None:
        time_range = [t_start, t_start + np.timedelta64(1, 'm')]
    


    time_label = W.Label(value=f'{time_range[0]}')

    time_slider = W.IntSlider(
        orientation='horizontal',
        description='time:',
        value=time_range[0],
        min=t_start,
        max=t_stop,
        readout=False,
        continuous_update=False,
        layout=W.Layout(width=f'{width_cm}cm')
    )

    delta = np.diff(time_range) / 1e9
    window_sizer = W.BoundedFloatText(value=delta, step=0.1, min=0.005, description='win (s)')


    main_widget = W.HBox([time_slider, window_sizer])
    some_widgets = {"time_label": time_label, "time_slider": time_slider, "window_sizer" : window_sizer}

    # return widget, controller
    return main_widget, some_widgets

def make_channel_selector(ds, width_cm=10, height_cm=1):
    channels = list(ds.keys())

    channel_selector = W.SelectMultiple(
        options=channels,
        value=channels,
        disabled=False,
        layout=W.Layout(width=f'{width_cm}cm', height=f'{height_cm}cm')
    )

    some_widgets = {'channel_selector': channel_selector}
    return channel_selector, some_widgets


class PlotUpdater:
    def __init__(self, ds, widgets, fig):
        self.ds = ds
        self.widgets = widgets
        self.fig = fig


        self.axs = None

    def __call__(self, change):
        print(change)

    def update_time(self, change):
        if self.axs is None:
            self.update_channel_visibility(None)
        
        
        t0 = self.widgets['time_slider'].value
        t1 = t0 + int(self.widgets['window_sizer'].value * 1e9)

        t0 = np.datetime64(t0, 'ns')
        t1 = np.datetime64(t1, 'ns')
        print(t0, t1)

        channels = self.get_visible_channels()

        for i, chan in enumerate(channels):
            ax = self.axs[i]
            ax.clear()

            d = {f'times_{chan}': slice(t0, t1)}
            sig = self.ds[chan].sel(**d)
            print(sig.shape)
            if sig.ndim > 1:
                continue
            print(sig)
            sig.plot.line(ax=ax)
            



        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    
    def update_channel_visibility(self, change):
        
        channels = self.get_visible_channels()
        self.fig.clear()
        n = len(channels)
        gs = self.fig.add_gridspec(n, 1)
        self.axs = [self.fig.add_subplot(gs[i]) for i in range(n)]

    def get_visible_channels(self):
        channels = self.widgets['channel_selector'].value
        return channels

cm = 1 / 2.54

def get_viewer(ds, width_cm=10, height_cm=8):

    ratios = [0.1, 0.8, 0.2]
    height_cm = width_cm * ratios[1]

    all_widgets = {}

    # figure
    with plt.ioff():
        output = W.Output()
        with output:
            fig, ax = plt.subplots(figsize=(width_cm * cm, height_cm * cm))

            plt.show()

    # time slider
    time_slider, some_widgets = make_timeslider(ds, width_cm=width_cm)
    all_widgets.update(some_widgets)

    # channels
    channel_selector, some_widgets = make_channel_selector(ds)
    all_widgets.update(some_widgets)


    updater = PlotUpdater(ds, all_widgets, fig)
    all_widgets['time_slider'].observe(updater.update_time)
    all_widgets['channel_selector'].observe(updater.update_channel_visibility)



    main_widget = W.AppLayout(
                center=fig.canvas,
                footer=time_slider,
                left_sidebar=None,
                right_sidebar=channel_selector,
                pane_heights=[0, 6, 1],
                pane_widths=ratios
            )


    return main_widget






