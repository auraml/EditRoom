"""
Constants for tools module functions.

This file contains all magic numbers and thresholds used in tools utilities,
making them easy to find, modify, and maintain in one central location.
"""

# Distance thresholds for movement commands
DISTANCE_THRESHOLD_OBVIOUS = 1.0  # meters - movements above this are "obviously" far
DISTANCE_THRESHOLD_SLIGHT = 0.5   # meters - movements below this are "slightly" close

# Angle thresholds for rotation commands  
ANGLE_THRESHOLD_OBVIOUS = 135     # degrees - rotations above this are "obvious"
ANGLE_THRESHOLD_SLIGHT = 45       # degrees - rotations below this are "slight"

# Scale thresholds for resizing commands
SCALE_THRESHOLD_OBVIOUS_UP = 1.3    # scale factor - enlargements above this are "obvious" 
SCALE_THRESHOLD_OBVIOUS_DOWN = 0.7  # scale factor - shrinking below this is "obvious"

# API configuration
API_RETRY_COUNT = 3               # number of retry attempts for API calls

# Collision detection parameters
MAX_COLLISION_RESOLUTION_ATTEMPTS = 100  # max attempts to resolve collisions
COLLISION_SEPARATION_FACTOR = 0.6        # factor for separating colliding objects
COLLISION_SAFETY_MARGIN = 0.1            # safety margin to prevent new collisions

# Default values
DEFAULT_ZERO_ANGLE = 0.0         # default angle when z_angle is missing 