import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def autoscale_turned_off(ax=None):
  "Suppresses pyplot's autoscale feature when included with a `with` statement"
  ax = ax or plt.gca()
  lims = [ax.get_xlim(), ax.get_ylim()]
  yield
  ax.set_xlim(*lims[0])
  ax.set_ylim(*lims[1])