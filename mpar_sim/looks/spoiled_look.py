from mpar_sim.looks.look import Look

class SpoiledLook(Look):
  """
  A look representing a beam that is spoiled on transmit but uses the full aperture on receive. This reduces the gain on transmit, but allows for full gain and angular resolution on receive.
  """