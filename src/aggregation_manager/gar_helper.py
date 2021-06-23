# Copyright (c) Anish Acharya.
# Licensed under the MIT License

from .mean import Mean
from .median import GeometricMedian, CoordinateWiseMedian
from .trimmed_mean import TrimmedMean
from .krum import Krum
from .norm_clipping import NormClipping
from typing import Dict
import numpy as np


def get_gar(aggregation_config: Dict):
    gar = aggregation_config.get("gar", 'mean')
    print('--------------------------------')
    print('Initializing {} GAR'.format(gar))
    print('--------------------------------')
    if gar == 'mean':
        return Mean(aggregation_config=aggregation_config)
    elif gar == 'geo_med':
        return GeometricMedian(aggregation_config=aggregation_config)
    elif gar == 'co_med':
        return CoordinateWiseMedian(aggregation_config=aggregation_config)
    elif gar == 'norm_clip':
        return NormClipping(aggregation_config=aggregation_config)
    elif gar == 'krum':
        return Krum(aggregation_config=aggregation_config)
    elif gar == 'trimmed_mean':
        return TrimmedMean(aggregation_config=aggregation_config)
    else:
        raise NotImplementedError
