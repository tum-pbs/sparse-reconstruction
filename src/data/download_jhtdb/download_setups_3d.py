from typing import Tuple

import numpy as np
import pyJHTDB.dbinfo as dbinfo
import math

def get_setup(name)-> Tuple[dict, dict, str]:
    if name == "channel":
        return get_channel()
    elif name == "isotropic1024coarse":
        return get_isotropic1024coarse()
    elif name == "mhd1024":
        return get_mhd1024()
    elif name == "transition_bl":
        return get_transition_bl()
    else:
        raise ValueError("Invalid simulation name")

# NOTE: JHTDB uses indexing starting from 1 for spatial and temporal dimensions
def get_channel():
    info = dbinfo.channel

    p = {
        "PDE": "Navier-Stokes: Channel Flow",
        "Dimension": 3,
        "Fields Scheme": "VVVp",
        "Fields": ["Velocity X", "Velocity Y", "Velocity Z", "Pressure"],
        "Reynolds Number": 1000,
        "Domain Extent (full)": [8*math.pi, 2, 3*math.pi],
        "Resolution (full)": [2048, 512, 1536],
        "Time Steps (full)": 4000,
        "Dt (full)": 0.0065,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive", "z negative", "z positive"],
        "Boundary Conditions": ["open", "open", "wall", "wall", "open", "open"],
        "Spatial Start": [1, 1, 1],
        "Spatial End": [512, 512, 512],
        "Spatial Step": [1, 1, 1],
        "Temporal Start": 1,
        "Temporal End": 4000,
        "Temporal Step": 10,
        "Constants": [],
    }
    p["Time Steps"] = (p["Temporal End"] - p["Temporal Start"] + 1) // p["Temporal Step"]
    p["Dt"] = p["Dt (full)"] * p["Temporal Step"]

    res = []
    dom = []
    for i in range(3):
        r = (p["Spatial End"][i] - p["Spatial Start"][i] + 1) // p["Spatial Step"][i]
        res += [r]
        dom += [p["Domain Extent (full)"][i] * (float(r) / p["Resolution (full)"][i])]
    p["Resolution"] = res
    p["Domain Extent"] = dom

    field_code = "up"
    return info, p, field_code


def get_isotropic1024coarse():
    info = dbinfo.isotropic1024coarse

    p = {
        "PDE": "Navier-Stokes: Isotropic Turbulence",
        "Dimension": 3,
        "Fields Scheme": "VVVp",
        "Fields": ["Velocity X", "Velocity Y", "Velocity Z", "Pressure"],
        "Reynolds Number": 433,
        "Domain Extent (full)": [2*math.pi, 2*math.pi, 2*math.pi],
        "Resolution (full)": [1024, 1024, 1024],
        "Time Steps (full)": 5028,
        "Dt (full)": 0.002,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive", "z negative", "z positive"],
        "Boundary Conditions": ["open", "open", "open", "open", "open", "open"],
        "Spatial Start": [1, 1, 1],
        "Spatial End": [512, 512, 512],
        "Spatial Step": [1, 1, 1],
        "Temporal Start": 1,
        "Temporal End": 5000,
        "Temporal Step": 10,
        "Constants": [],
    }
    p["Time Steps"] = (p["Temporal End"] - p["Temporal Start"] + 1) // p["Temporal Step"]
    p["Dt"] = p["Dt (full)"] * p["Temporal Step"]

    res = []
    dom = []
    for i in range(3):
        r = (p["Spatial End"][i] - p["Spatial Start"][i] + 1) // p["Spatial Step"][i]
        res += [r]
        dom += [p["Domain Extent (full)"][i] * (float(r) / p["Resolution (full)"][i])]
    p["Resolution"] = res
    p["Domain Extent"] = dom

    field_code = "up"
    return info, p, field_code


def get_mhd1024():
    info = dbinfo.mhd1024

    p = {
        "PDE": "Magnetohydrodynamic Turbulence",
        "Dimension": 3,
        "Fields Scheme": "VVVpMMMOOO",
        "Fields": [
            "Velocity X",
            "Velocity Y",
            "Velocity Z",
            "Pressure",
            "Magnetic Field X",
            "Magnetic Field Y",
            "Magnetic Field Z",
            "Vector Potential X",
            "Vector Potential Y",
            "Vector Potential Z",
        ],
        "Reynolds Number": 186,
        "Domain Extent (full)": [2*math.pi, 2*math.pi, 2*math.pi],
        "Resolution (full)": [1024, 1024, 1024],
        "Time Steps (full)": 1024,
        "Dt (full)": 0.0025,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive", "z negative", "z positive"],
        "Boundary Conditions": ["open", "open", "open", "open", "open", "open"],
        "Spatial Start": [1, 1, 1],
        "Spatial End": [512, 512, 512],
        "Spatial Step": [1, 1, 1],
        "Temporal Start": 1,
        "Temporal End": 1000,
        "Temporal Step": 10,
        "Constants": [],
    }
    p["Time Steps"] = (p["Temporal End"] - p["Temporal Start"] + 1) // p["Temporal Step"]
    p["Dt"] = p["Dt (full)"] * p["Temporal Step"]

    res = []
    dom = []
    for i in range(3):
        r = (p["Spatial End"][i] - p["Spatial Start"][i] + 1) // p["Spatial Step"][i]
        res += [r]
        dom += [p["Domain Extent (full)"][i] * (float(r) / p["Resolution (full)"][i])]
    p["Resolution"] = res
    p["Domain Extent"] = dom

    field_code = "upba"
    return info, p, field_code


def get_transition_bl():
    info = dbinfo.transition_bl

    p = {
        "PDE": "Navier-Stokes: Transitional Boundary Layer",
        "Dimension": 3,
        "Fields Scheme": "VVVp",
        "Fields": ["Velocity X", "Velocity Y", "Velocity Z", "Pressure"],
        "Reynolds Number": 800,
        "Domain Extent (full)": [969.8465, 26.4844, 240],
        "Resolution (full)": [10240, 1536, 2048], # NOTE: only the [0:3320, 0:224, 0:2048] slice actually contains useable data
        "Time Steps (full)": 4701,
        "Dt (full)": 0.25,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive", "z negative", "z positive"],
        "Boundary Conditions": ["open", "open", "wall", "open", "open", "open"],
        "Spatial Start": [3096, 1, 1],
        "Spatial End": [3319, 224, 224],
        "Spatial Step": [1, 1, 1],
        "Temporal Start": 1,
        "Temporal End": 4700,
        "Temporal Step": 5,
        "Constants": [],
    }
    p["Time Steps"] = (p["Temporal End"] - p["Temporal Start"] + 1) // p["Temporal Step"]
    p["Dt"] = p["Dt (full)"] * p["Temporal Step"]

    res = []
    dom = []
    for i in range(3):
        r = (p["Spatial End"][i] - p["Spatial Start"][i] + 1) // p["Spatial Step"][i]
        res += [r]
        dom += [p["Domain Extent (full)"][i] * (float(r) / p["Resolution (full)"][i])]
    p["Resolution"] = res
    p["Domain Extent"] = dom

    field_code = "up"
    return info, p, field_code