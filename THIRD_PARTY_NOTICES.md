# Third-party notices

## CybORG (CAGE Challenge 4 simulator), `CybORG/`

Copyright (c) Commonwealth of Australia 2019. Developed by the Defence Science and Technology
Group (DST Group), with portions that are works of the United States Government released under
the CC0 1.0 Universal public domain dedication. Licensed under the MIT License.

The directory `CybORG/` is the upstream CAGE Challenge 4 release (`CybORG/version.txt` = 4.0),
vendored **unmodified**, including its `LICENSE.txt` and the per-file copyright headers. The
testbed never edits the simulator; the phase pinning used by the research layer is done by
subclassing the scenario generator in `flipthem/resources.py`.

Upstream project: the CAGE Challenge 4 public repository of the TTCP CAGE working group.

## Python dependencies

`requirements.txt` pins numpy, gym, gymnasium, networkx, PyYAML, prettytable and pygame, each
under its own license (BSD / MIT / LGPL for pygame). They are installed from PyPI and are not
redistributed here.
