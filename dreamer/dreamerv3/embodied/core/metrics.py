import collections
import warnings

import numpy as np


class Metrics:

  def __init__(self):
    self._scalars = collections.defaultdict(list)
    self._lasts = {}

  def scalar(self, key, value):
    self._scalars[key].append(value)

  def image(self, key, value):
    self._lasts[key].append(value)

  def video(self, key, value):
    self._lasts[key].append(value)

  def add(self, mapping, prefix=None):
    for key, value in mapping.items():
      key = prefix + '/' + key if prefix else key
      if hasattr(value, 'shape') and len(value.shape) > 0:
        self._lasts[key] = value
      else:
        self._scalars[key].append(value)

  def result(self, reset=True):
    result = {}
    result.update(self._lasts)
    with warnings.catch_warnings():  # Ignore empty slice warnings.
      warnings.simplefilter('ignore', category=RuntimeWarning)
      for key, values in self._scalars.items():
        result[key] = np.nanmean(values, dtype=np.float64)
    reset and self.reset()
    return result

  def reset(self):
    self._scalars.clear()
    self._lasts.clear()
    
  def get_key(self, key):
    
    if key in self._lasts.keys():
      return self._lasts[key]
    elif key in self._scalars.keys():
      return self._scalars[key]
    else:
      print(f"{key} not in metric returning None")
      return None
    
  def clear_key(self, key):
    self._scalars[key].clear()
    
  def get_metric_keys(self):    
    return [*self._lasts.keys(), *self._scalars.keys()]
      
