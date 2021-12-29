# Business card renderer in python

This code implements the ray casting and creates this animation using _python_ + _numba_ for speed.

![](animation.gif)

I described my journey writing this code in details here:
- part 1, intro - https://medium.com/@sibearianpython/the-business-card-raytracer-in-python-8dcc868069f5
- part 2, animation - https://medium.com/@sibearianpython/the-business-card-raytracer-in-python-part-2-58fd490c17f7

The idea to make a ray caster in python is inspired by this post - http://fabiensanglard.net/rayTracing_back_of_business_card/

## Installation
TL;DR: To install, create a new virtual environment and install the required dependencies by calling:
```bash
python -m venv /path/to/new/venv
source /path/to/new/venv/bin/activate
# for Windows: C:\path\to\new\venv\Scripts\activate.bat
pip install -r requirements.txt
```

The code was written in python3.6.3 (32 bit).
It _should_ work in the later versions too (code inspection in PyCharm told me that it has no compatibility issues up to
and including 3.10), but I didn't test it.

The installation code only creates a new clean virtual environment and 
installs the proper versions of three packages: _numpy_, _numba_, and _opencv-python_. You can do it manually
(or skip altogether if you already have the right versions of these packages).

## Running
```bash
python business_card_renderer.py
```
and wait (number of rendered frame will be printed each time rendering is complete).
By default, only 10 frames will be rendered to spare you waiting time. 
You can change it on top of the script to 120 if you want to see the entire animation.

Note: the window with the animation may not show up on top.

To close the window with the animation press "Esc".
