# Coronagraph (YIP) Guide

pyEDITH models coronagraphs using `CoronagraphYIP`, a wrapper around
[yippy](https://yippy.readthedocs.io)'s `Coronagraph` class. `yippy` handles
loading the Yield Input Package (YIP) FITS files and computing performance
curves (throughput, contrast, core area). `CoronagraphYIP` extracts the
quantities pyEDITH needs for exposure time calculations: the noise floor,
stellar intensity maps, and the off-axis PSF.

## Two ways to initialize

`CoronagraphYIP` can be initialized with either a **path** to a YIP directory
or a **pre-constructed** `yippy.Coronagraph` object:

```python
from pyEDITH.components.coronagraphs import CoronagraphYIP

# From a YIP directory (yippy loads FITS files internally)
coronagraph = CoronagraphYIP(path="/path/to/yip")

# From a pre-constructed yippy.Coronagraph
from yippy import Coronagraph
coro = Coronagraph("/path/to/yip", psf_trunc_ratio=0.3)
coronagraph = CoronagraphYIP(yippy_coro=coro)
```

Both produce identical results. The difference is performance when called
repeatedly.

## Best practices for loops

When calling pyEDITH in a loop (e.g. sweeping over a target catalog)
the path-based approach reconstructs the `yippy.Coronagraph` from FITS on every
call:

```python
for target_params in catalog:
    coronagraph = CoronagraphYIP(path="/path/to/yip")
    # ~250 ms per call (FITS I/O + performance curve computation)
```

Pre-loading the coronagraph once and reusing it avoids this overhead:

```python
from yippy import Coronagraph

coro = Coronagraph("/path/to/yip", psf_trunc_ratio=0.3)

for target_params in catalog:
    coronagraph = CoronagraphYIP(yippy_coro=coro)
    # ~21 ms per call (skips FITS I/O entirely)
```

**Speedup: ~12x per call.** Both approaches produce identical exposure times.

```{note}
To switch `psf_trunc_ratio` mid-sweep without reloading from disk, use
`coro.set_psf_trunc_ratio(new_ratio)`. Results are cached to
`yippy_cache/performance/` for reuse.
```
