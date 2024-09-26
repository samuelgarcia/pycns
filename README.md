# pycns

Reader and tools for analysing CNS monitor data

## Doc

https://pycns.readthedocs.io


## Installation

```bash
git clone https://github.com/samuelgarcia/pycns.git
cd pycns
pip install -e .
```

Update:
```bash
cd pycns
git clone pull origin main
```

## Example


```python
from pycns import CnsReader, get_viewer
from pathlib import Path

raw_folder = Path('/XXX/YYY/ZZZ/')

# this shows all available streams
cns_reader = CnsReader(raw_folder)
print(cns_reader)

# this shows stream object
print(cns_reader.streams)

# get some chunk with time vector handled with numpy.datetime64
sig, times = cns_reader.streams['CO2'].get_data(isel=slice(100_000, 110_000), with_times=True, apply_gain=True)

# easy viewer to navigate (this works only in jupyter)
viewer = get_viewer(cns_reader)
display(viewer)

# export some streams to xarray with a resample on common time base
stream_names = ['ECG_II', 'RESP', 'EEG']
start = '2021-01-08T00:10:15'
stop = '2021-01-08T00:30:52'
ds = cns_reader.export_to_xarray(stream_names, start=start, stop=stop, resample=True, sample_rate=100.)
```


Note : 
  * to get the viewer (based on ipywidgets) properly in vscode you need `pip install -U ipywidgets==7.7.1`