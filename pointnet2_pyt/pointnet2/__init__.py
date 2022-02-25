'''
Description: 
Autor: Jiachen Sun
Date: 2022-02-16 22:23:16
LastEditors: Jiachen Sun
LastEditTime: 2022-02-24 23:12:32
'''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

__version__ = "2.1.1"

try:
    __POINTNET2_SETUP__
except NameError:
    __POINTNET2_SETUP__ = False

if not __POINTNET2_SETUP__:
    from pointnet2_pyt.pointnet2 import utils
    from pointnet2_pyt.pointnet2 import data
    from pointnet2_pyt.pointnet2 import models
