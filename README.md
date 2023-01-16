# pycns
Reader and tools for analysing CNS monitor data


Installation:

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

Example convert raw to xarray:
```python
from pycns import convert_folder_to_dataset
from pathlib import Path

base_folder = Path('/XXX/YYY/ZZZ')
raw_folder = base_folder / 'raw_data/P1'
zarr_folder = base_folder / 'data_xarray/P1'
    
load_streams = ['EEG', 'RESP', 'ECG']
convert_folder_to_dataset(raw_folder, zarr_folder, load_streams=load_streams, progress_bar=False)
```

Example resample all stream to unique to unique sample rate:
```python
from pycns import convert_folder_to_dataset
from pathlib import Path

base_folder = Path('/XXX/YYY/ZZZ')
source_folder = base_folder / 'data_xarray/P1'
target_folder = base_folder / 'data_xarray/P1_resample'

resample(source_folder, target_folder, sample_rate=200., stream_names=['ECG', 'RESP', ])
```


Example open viewer in jupyter:
```python
%matplotlib widget

base_folder = Path('/XXX/YYY/ZZZ')
dataset_folder = base_folder / 'data_xarray' / 'P1'
ds = xr.open_zarr(dataset_folder)
ds


main_widget = get_viewer(ds)
main_widget
```