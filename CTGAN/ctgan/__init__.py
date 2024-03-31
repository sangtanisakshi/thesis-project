# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '0.9.1.dev0'

from ctgan.demo import load_demo
from ctgan.synthesizers.ctgan_m import CTGAN
from ctgan.synthesizers.tvae import TVAE

__all__ = (
    'CTGAN',
    'TVAE',
    'load_demo'
)
