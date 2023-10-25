from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
from tbparse import SummaryReader
from mpar_sim.interference.recorded import RecordedInterference

if __name__ == '__main__':
  plt.rcParams['font.size'] = 14
  plt.rcParams['font.weight'] = 'bold'
  
  filename = "/home/shane/data/HOCAE_Snaps_bool.dat"
  order = 'F'
  interf = RecordedInterference(filename, order=order)
  interf.data[:, 512] = 0
  interf.data[:, 100:105] = 0
  interf.data[:, -105:-100] = 0
  plt.imshow(interf.data[9000:9512], aspect='auto', origin='lower', extent=(-100, 100, 0, 10))
  plt.xlabel('Baseband Frequency (MHz)', fontsize=14, fontweight='bold')
  plt.ylabel('Time (ms)', fontsize=14, fontweight='bold')
  plt.colorbar()
  plt.savefig('./2_45_unoccupied.pdf', bbox_inches='tight')