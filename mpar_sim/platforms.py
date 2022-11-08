import numpy as np

class Platform():
    id = 0
    
    def __init__(self,
                 position: np.ndarray = np.zeros((3,)),
                 velocity: np.ndarray = np.zeros((3,)),
                 acceleration: np.ndarray = np.zeros((3,)),
                 orientation: np.ndarray = np.zeros((3,)),
                 angular_velocity: np.ndarray = np.zeros((3,)),
                 rcs: float = 1,
                 ):
        
        
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.rcs = rcs
        self.id = Platform.id
        Platform.id += 1
        
        
    def update(self, dt: float) -> None:
        """
        Update the position based on the current velocity and acceleration
        
        Args:
            dt (float): Time increment
        """
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2
        self.velocity += self.acceleration * dt

        
        