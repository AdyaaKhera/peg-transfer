# Third-party notices

This research project uses the following third-party robots, software, models and materials. None of them are
included in this repository except where noted; each stays under its own owner's terms.

## I2RT YAM arms (robot model)

The two robot arms are I2RT Robotics' YAM arms. The robot description (URDF and meshes,
`yam_station_linear_4310_d405`) comes from the I2RT repository, <https://github.com/i2rt-robotics/i2rt>, and is not
copied here. One file in this repository is derived from it: `peg_transfer_benchmark/core/yam_geometry.npz` (surface
points and finger measurements sampled from the I2RT meshes by `scripts/setup/robot_geometry.py`). It is used under
I2RT's MIT License:

```
MIT License

Copyright (c) I2RT Robotics

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

YAM and I2RT are names of I2RT Robotics. This project is not affiliated with or endorsed by I2RT Robotics.

## NVIDIA Isaac Sim 6.0.1 (simulator)

The simulation runs in NVIDIA Isaac Sim 6.0.1, <https://developer.nvidia.com/isaac/sim>. Its source code is released
under the Apache License 2.0, and the Omniverse components it needs to run are under the NVIDIA Isaac Sim Additional
Software and Materials License. Isaac Sim is not included in this repository; the code here only calls its Python API
and must be run in your own Isaac Sim installation. The videos and images in `fls_writeup_media/` were rendered with
Isaac Sim. NVIDIA, Omniverse and Isaac Sim are trademarks of NVIDIA Corporation.

## AI models

- **GPT-6.1 Sol** by OpenAI, used through the OpenAI API under OpenAI's terms.
- **Claude Opus 5.5** by Anthropic, used through the OpenRouter API (<https://openrouter.ai>) under Anthropic's and
  OpenRouter's terms.

The model outputs quoted in the write-up (reasoning, tool calls, lessons) come from these models. No API keys are
included in this repository.

## Python libraries

Used at run time and installed separately (not included here): NumPy, SciPy and Pillow, matplotlib for the plots,
imageio and imageio-ffmpeg (FFmpeg) for the videos, and Pixar's USD (`pxr`) and Omniverse modules, which ship with
Isaac Sim. Each is under its own open-source license.

## Development tools

Much of the code and documentation was written with Claude Code (Anthropic).

## FLS

FLS (Fundamentals of Laparoscopic Surgery) is a program of SAGES and the American College of Surgeons. This project
only reproduces the peg transfer task in simulation for research; it is not affiliated with or endorsed by SAGES, the
American College of Surgeons or the FLS program, and it is not an FLS test.

## Reference

The automated assessment of the peg transfer that the write-up cites [1]: F. Rashidi Fathabadi, J. L. Grantner, S. A.
Shebrain, and I. Abdel-Qader, "Autonomous sequential surgical skills assessment for the peg transfer task in a
laparoscopic box-trainer system with three cameras," *Robotica*, vol. 41, no. 6, pp. 1837-1855, 2023.
doi: [10.1017/S0263574723000218](https://doi.org/10.1017/S0263574723000218). No material from the paper is copied here.
