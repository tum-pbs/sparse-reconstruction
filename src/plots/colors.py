import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Sequence
import seaborn as sns

BASECOLOR=['#FF1F5B', '#009ADE', '#FFC61E', '#AF58BA', '#F28522', '#00CD6C','#A6761D']

def generate_colormap_from_list(color_list: Sequence,name:str,qualitative:bool=False)->colors.Colormap:
    if qualitative:
        return colors.ListedColormap(color_list,name=name)
    else:
        return colors.LinearSegmentedColormap.from_list(name,color_list)
    
def colors_from_list(n_colors:int,color_list=BASECOLOR)-> list:
    if n_colors <= len(color_list):
        return color_list[:n_colors]
    else:
        return generate_colormap_from_list(color_list,'colors_from_list_temp',qualitative=False)(np.linspace(0,1,n_colors,endpoint=True))

def colors_from_colormap(cmap_name:str,n_colors:int)->list:
    return plt.cm.get_cmap(cmap_name)(np.linspace(0,1,n_colors,endpoint=True))

CASES=[
    "fff-s",
    "scot",
    "fno",
    "galekin",
    "oformer",
    "faster-dit",
    "fff-b",
    "fff-l",
    "fff-v1",
    "unet",
]


FFFCOLORS={
    key:color for key,color in zip(CASES,colors_from_list(len(CASES),sns.color_palette(as_cmap=True)))
    #key:color for key,color in zip(CASES,colors_from_list(len(CASES)))
    #key:color for key,color in zip(CASES,colors_from_colormap("rainbow",len(CASES)))
}