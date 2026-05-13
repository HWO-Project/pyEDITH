# Coronagraph (YIP) Guide

During the LUVOIR and HabEx studies a standard "handshake" between coronagraph
designers and yield modelers was created called a "Yield Input Package" or YIP
([see the definition here](https://starkspace.com/yield_standards.pdf)).
pyEDITH uses YIPs to represent coronagraphs and compute the necessary
performance metrics.

## Modeling the coronagraph

A coronagraphic exposure time calculation needs various
quantities to represent the coronagraph, primarily: 
- How much planet light reaches the detector
- How big a photometric aperture to integrate it over
- How much stellar leakage contaminates that aperture
- How much extended-source emission (zodi, exozodi) gets through.
All four of those quantities are calculated from the same underlying FITS files that
compose a YIP.

To avoid repeated code pyEDITH uses [yippy](https://yippy.readthedocs.io) to
read YIPs and calculate [coronagraph performance
metrics](https://yippy.readthedocs.io/en/latest/examples/00_Performance_Metrics_Overview.html).
pyEDITH's `CoronagraphYIP` is a thin wrapper that calls yippy for the
quantities below and exposes them in the form pyEDITH's noise budget expects.

## Finding a YIP

We currently recommend that people download from the catalog of YIPs hosted on Zenodo while a long
term hosting solution is determined. These can be downloaded with a single call. Browse
the catalog with `yippy.list_yips()`, or view the rendered table on the
[yippy datasets page](https://yippy.readthedocs.io/en/latest/datasets.html).
To pull a specific YIP, pass its catalog name to `fetch_yip`, which
downloads the archive on first call and serves it from a local cache on
subsequent calls.

```python
from yippy import list_yips, fetch_yip

list_yips()                  # all available names
list_yips(telescope="eac1")  # narrow by telescope, coronagraph, or sampling
yip_path = fetch_yip("eac1_aavc_2d")
```

For in-development YIPs that are not yet on Zenodo, drop the YIP folder
somewhere on disk and point the `YIP_CORO_DIR` environment variable at
it. pyEDITH will pick it up when a registry entry refers to a folder
name instead of a catalog name (see [Where the YIP comes
from](#where-the-yip-comes-from) below).

## Coronagraph quantities in the ETC

These yippy outputs feed directly into pyEDITH's exposure time
calculation. Each link below points to the relevant documentation
page for definitions and examples.

- **Photometric aperture throughput** (`yippy_obj.throughput_map()`):
  fraction of planet light captured in the photometric aperture at each
  separation, set by `psf_trunc_ratio` (default 0.3 captures the
  off-axis PSF down to 30% of its peak). See
  [Core Throughput](https://yippy.readthedocs.io/en/latest/examples/01_Core_Throughput.html).
- **Photometric aperture solid angle**,
  {math}`\Omega` (`yippy_obj.core_area_map()`): size of the aperture,
  in {math}`(\lambda/D)^2`. Also set by `psf_trunc_ratio`, and used to
  convert background surface brightness (zodi, exozodi) into a count
  rate. See
  [Spatial Metrics and Backgrounds](https://yippy.readthedocs.io/en/latest/examples/03_Spatial_Metrics_and_Backgrounds.html).
- **Stellar intensity at the planet location**,
  {math}`I_*` (`yippy_obj.core_mean_intensity_map(stellar_diam)` for
  the fast radially-averaged form, or `yippy_obj.stellar_intens(stellar_diam)`
  for the full 2D map): residual starlight leaking through the
  coronagraph, which sets the stellar speckle background. See
  [Stellar Leakage and Contrast](https://yippy.readthedocs.io/en/latest/examples/02_Stellar_Leakage_and_Contrast.html).
- **Sky transmission map** (`yippy_obj.sky_trans()`): throughput for
  extended emission as a function of position, multiplied into zodi and
  exozodi count rates so they pick up the coronagraph mask geometry.
- **Separation grid** (`yippy_obj.separation_map()`): radial separation
  in {math}`\lambda/D` at every pixel, used to evaluate the maps above
  at the planet's position.
- **YIP header geometry** (`yippy_obj.header`): pixel scale, image
  size, and occulter center. Used to map between pixel space and
  {math}`\lambda/D` for the rest of the pipeline.

`CoronagraphYIP` exposes these as numpy arrays on the configured
observatory, and from that point on the exposure time calculator does
not need to know that yippy exists.

## Recommended pattern: load once, pass everywhere

The single biggest performance win is to construct a
`yippy.Coronagraph` **once** and reuse it for every pyEDITH call.
Re-loading a YIP from FITS for every target costs ~250 ms per call.
Reusing a pre-loaded `Coronagraph` is ~21 ms per call (a ~12x speedup),
and the exposure time is identical.

```python
from yippy import Coronagraph, fetch_yip
from pyEDITH.components.coronagraphs import CoronagraphYIP

# Fetch the YIP (downloaded from Zenodo on first call, cached after).
# See https://yippy.readthedocs.io/en/latest/datasets.html for catalog.
yip_path = fetch_yip("eac1_aavc_2d")

# Build the yippy.Coronagraph once.
coro = Coronagraph(yip_path, psf_trunc_ratio=0.3)

# Pass it into pyEDITH for every target.
for target_params in catalog:
    coronagraph = CoronagraphYIP(yippy_coro=coro)
    # ... run pyEDITH using `coronagraph` ...
```

To switch `psf_trunc_ratio` mid-sweep without reloading from disk:

```python
coro.set_psf_trunc_ratio(0.5)
```

Performance-curve results are cached to `yippy_cache/performance/` and
reused on subsequent calls with the same ratio. The trade-offs across
different ratios are covered in
[Aperture Methods Comparison](https://yippy.readthedocs.io/en/latest/examples/05_Aperture_Methods_Comparison.html).

## Other ways to initialize

`CoronagraphYIP` can also be initialized with a path to a YIP
directory, which is convenient for one-shot calls but reloads the YIP
from FITS each time.

```python
# Catalog name -- yippy downloads + caches the YIP for you.
coronagraph = CoronagraphYIP(path=fetch_yip("eac1_aavc_2d"))

# A folder on disk (e.g. an in-development YIP not yet on Zenodo).
coronagraph = CoronagraphYIP(path="/path/to/yip")
```

For loops, prefer the pre-constructed `Coronagraph` pattern shown
above.

## Where the YIP comes from

When pyEDITH is told to use a registry entry (for example via
`observatory_preset="EAC1"`), the loader looks at the entry's fields
to decide how to resolve the YIP:

1. **`yippy_name`** (preferred). The YIP is fetched from Zenodo via
   `yippy.fetch_yip` and cached locally. No environment variables
   required. Browse names with `yippy.list_yips()` or see the
   [yippy datasets page](https://yippy.readthedocs.io/en/latest/datasets.html).
2. **`path`** (legacy). The folder name is joined with the
   `YIP_CORO_DIR` environment variable. Use this for in-development
   YIPs that have not yet been published to Zenodo.

See [Advanced Options](advanced.md#add-your-own-yip-coronagraph) for
the registry schema.
