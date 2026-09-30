# sscbirrt-assets

The [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) UR5e and Robotiq 2F-85 models, packaged so that `pip install "sscbirrt[demo]"` can run the [sscbirrt](https://github.com/personalrobotics/sscbirrt) demos without cloning the Menagerie.

```python
import sscbirrt_assets

sscbirrt_assets.ur5e_xml()          # .../menagerie/universal_robots_ur5e/ur5e.xml
sscbirrt_assets.robotiq_2f85_xml()  # .../menagerie/robotiq_2f85/2f85.xml
sscbirrt_assets.menagerie_path()    # a directory laid out like a Menagerie clone
```

The functions return real file paths, since MJCF loaders resolve meshes relative to the XML.

## Provenance and licenses

Each file is copied unmodified from Menagerie at the commit in `sscbirrt_assets.MENAGERIE_COMMIT` and checked against the SHA-256 in `manifest.json`. Only the MJCF, the meshes and the LICENSE of each model ship.

- UR5e: BSD-3-Clause, Copyright 2018 ROS Industrial Consortium
- Robotiq 2F-85: BSD-2-Clause, Copyright (c) 2013, ROS-Industrial
- The Python code in this package: MIT

## Building

`fetch.py` places the files. The build hook calls it, so `uv build assets` works on its own. Pass `python assets/fetch.py --from <clone>` to copy from a local Menagerie clone instead of downloading. Either way the hashes are checked. After moving the pin, regenerate the manifest with `--write-manifest --from <clone>`.
