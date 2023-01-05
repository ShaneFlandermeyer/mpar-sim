class KalmanUpdater():
  def __init__(self,
               measurement_model) -> None:
    self.measurement_model = measurement_model
  
  def update(self, )