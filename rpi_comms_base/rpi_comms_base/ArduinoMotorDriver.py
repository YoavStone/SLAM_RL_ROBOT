class ArduinoMotorDriver:
    """
    A class that represents a motor controlled by Arduino.
    Similar interface to the original MotorDriver but without GPIO dependencies.
    """
    def __init__(self):
        # Initialize state variables
        self.pwm = 0.7  # 0-1 scale
        self.en_count = 0
        self.dir = 0

    def forward(self):
        """Set motor direction to forward."""
        self.dir = 1

    def backwards(self):
        """Set motor direction to backward."""
        self.dir = -1

    def stop(self):
        """Stop the motor."""
        self.dir = 0
        self.pwm = 0.0

    def change_pwm(self, pwm):
        """Update the PWM value."""
        self.pwm = min(pwm, 1.0)

    def update_encoder_count(self, count):
        """Update encoder count value."""
        self.en_count = count

    def get_pos(self):
        """Return the current encoder position."""
        return self.en_count

    def reset(self):
        self.en_count = 0