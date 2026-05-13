# Advanced Options

## Change a preset

`pyEDITH` has some standard presets for the currently assumed Exploratory Analytic Cases for HWO. If you need to specify a different preset (e.g. changing the coronagraph), you can replace this parameter keyword that we use in the [Imaging Tutorial](imaging_tutorial.ipynb) and the [Spectroscopy Tutorial](spectroscopy_tutorial.ipynb):


```python
parameters["observatory_preset"] = "EAC1"
```

with three keywords:

```python
parameters["telescope_type"] = "EAC1"
parameters["coronagraph_type"] = "MyNewCoronagraph"
parameters["detector_type"] = "EAC1"
```


## Add your own YIP coronagraph

Open `src/pyEDITH/components/registry.json`. This file connects your preferred keyword to a coronagraph YIP. Each coronagraph entry uses one of two fields to describe where the YIP comes from:

- **`yippy_name`** — a yippy catalog name (e.g. `eac1_aavc_2d`). The YIP is fetched from Zenodo on first use and cached locally. Browse names with `yippy.list_yips()` or see the [yippy datasets page](https://yippy.readthedocs.io/en/latest/datasets.html). This is the recommended way to add a coronagraph.
- **`path`** — a local folder name resolved against the `YIP_CORO_DIR` environment variable. Use this for in-development YIPs that are not yet on Zenodo.

Use exactly one of these fields per entry; setting both is rejected.

The registry starts out looking something like this:

```json
{
    "telescopes": {
        "EAC1": {
            "class": "EACTelescope",
            "path": ""
        }
    },
    "coronagraphs": {
        "LUVOIR": {
            "class": "CoronagraphYIP",
            "path": "usort_offaxis_ovc"
        }
    },
    "detectors": {
        "EAC1": {
            "class": "EACDetector",
            "path": ""
        }
    }
}
```

Add a new entry under `"coronagraphs"` with the class `CoronagraphYIP` and either `yippy_name` (preferred) or `path` (legacy):

```json
{
    "telescopes": {
        "EAC1": {
            "class": "EACTelescope",
            "path": ""
        }
    },
    "coronagraphs": {
        "LUVOIR": {
            "class": "CoronagraphYIP",
            "path": "usort_offaxis_ovc"
        },
        "MyZenodoCoro": {
            "class": "CoronagraphYIP",
            "yippy_name": "eac1_aavc_2d"
        },
        "MyLocalCoro": {
            "class": "CoronagraphYIP",
            "path": "NewCoronagraph"
        }
    },
    "detectors": {
        "EAC1": {
            "class": "EACDetector",
            "path": ""
        }
    }
}
```

`MyZenodoCoro` is auto-downloaded from Zenodo. `MyLocalCoro` is loaded from `$YIP_CORO_DIR/NewCoronagraph/`.

To use a new coronagraph, set the corresponding `coronagraph_type` keyword in your parameters (see [above](advanced.md#change-a-preset)).