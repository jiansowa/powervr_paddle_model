
import numpy as np


def default_collate_fn(batch_data):
    elem = batch_data[0]
    if isinstance(elem, np.ndarray):
        return np.array(batch_data, dtype=elem.dtype)
    elif isinstance(elem, float):
        return np.array(batch_data, dtype=np.float32)
    elif isinstance(elem, int):
        return np.array(batch_data)
    elif isinstance(elem, tuple):
        data = [[] for i in range(len(elem))]
        for i, d in enumerate(batch_data):
            for j in range(len(elem)):
                data[j].append(d[j])
        return [default_collate_fn(x) for x in data]

    raise TypeError("batch data must container ")


