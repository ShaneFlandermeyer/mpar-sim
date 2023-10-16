from mpar_sim.interference.interference import Interference
import numpy as np

class RecordedInterference(Interference):
  """Interference from recorded spectrum data"""
  def __init__(self,
               filename: str,
               fft_size: int = 1024,
               dtype=np.uint8,
               order='C'
               ) -> None:
    self.fft_size = fft_size
    
    self.data = np.fromfile(filename, dtype=dtype)
    self.data = self.data.reshape((-1, fft_size), order=order)
    self.n_snapshots = self.data.shape[0]
    
    self.reset()
    
  def step(self, time = None): 
    self.start_ind = (self.start_ind + 1) % self.n_snapshots
    self.state = self.data[self.start_ind]
    return self.state
  
  def reset(self):
    self.start_ind = np.random.randint(0, self.n_snapshots)
    self.state = self.data[self.start_ind]
    return self.state
    
    
  