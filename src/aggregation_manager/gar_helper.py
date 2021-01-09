# Copyright (c) Anish Acharya.
# Licensed under the MIT License

from .mean import Mean
from typing import Dict


def get_gar(aggregation_config: Dict):
    gar = aggregation_config.get("gar", 'mean')
    print('--------------------------------')
    print('Initializing {} GAR'.format(gar))
    print('--------------------------------')
    if gar == 'mean':
        return Mean(aggregation_config=aggregation_config)
    else:
        raise NotImplementedError
