class DistanceEstimator:
    """
    Estimates distance based on the pinhole camera model:
      distance_cm = (known_width_cm * focal_length) / pixel_width
    """
    def __init__(self, focal_length, known_width_cm):
        self.focal_length = focal_length
        self.known_width = known_width_cm

    def estimate(self, pixel_width):
        """
        Given the object width in pixels, return distance in cm.
        """
        if pixel_width <= 0:
            return None
        return (self.known_width * self.focal_length) / pixel_width
