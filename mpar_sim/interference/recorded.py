from mpar_sim.interference.interference import Interference
import numpy as np

class RecordedInterference(Interference):
  """Interference from recorded spectrum data"""
  def __init__(self,
               filename: str,
               fft_size: int = 1024,
               ) -> None:
    self.fft_size = fft_size
    
    self.data = np.fromfile(filename, dtype=np.uint8)
    self.data = self.data.reshape((-1, fft_size))
    self.n_snapshots = self.data.shape[0]
    
  def step(self, time): 
    self.start_ind = (self.start_ind + 1) % self.n_snapshots
    self.state = self.data[self.start_ind]
    return self.state
  
  def reset(self):
    self.start_ind = np.random.randint(0, self.n_snapshots)
    self.state = self.data[self.start_ind]
    return self.state
    
    
  