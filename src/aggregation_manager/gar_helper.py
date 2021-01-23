# Copyright (c) Anish Acharya.
# Licensed under the MIT License

from .mean import Mean
from .geomed import GeometricMedian
from .trimmed_mean import TrimmedMean
from typing import Dict


def get_gar(aggregation_config: Dict):
    gar = aggregation_config.get("gar", 'mean')
    print('--------------------------------')
    print('Initializing {} GAR'.format(gar))
    print('--------------------------------')
    if gar == 'mean':
        return Mean(aggregation_config=aggregation_config)
    elif gar == 'geo_med':
        return GeometricMedian(aggregation_config=aggregation_config)
    else:
        raise NotImplementedError
