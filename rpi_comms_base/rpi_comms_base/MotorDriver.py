from gpiozero import DigitalOutputDevice, PWMOutputDevice, RotaryEncoder


class MotorDriver:
    def __init__(self, pin1, pin2, pwm_pin, pwm_frequency, enable_pin_a, enable_pin_b):
        # Set up direction control pins
        self.in1 = DigitalOutputDevice(pin1)
        self.in2 = DigitalOutputDevice(pin2)

        # Set up PWM control pin (ENA/ENB on the L298N)
        self.enable_pin = PWMOutputDevice(pwm_pin, frequency=pwm_frequency)

        # Set up encoder pins as input devices
        self.encoder = RotaryEncoder(enable_pin_a, enable_pin_b, max_steps=0)

        # Initialize state variables
        self.pwm = 0.7  # 0-1 scale
        self.en_count = 0
        self.dir = 0

        # Set initial pwm
        self.enable_pin.value = self.pwm

    def forward(self):
        # Set direction pins
        self.in1.on()  # HIGH
        self.in2.off()  # LOW
        # Ensure pwm is applied
        self.enable_pin.value = self.pwm
        self.dir = 1

    def backwards(self):
        # Set direction pins
        self.in1.off()  # LOW
        self.in2.on()  # HIGH
        # Ensure pwm is applied
        self.enable_pin.value = self.pwm
        self.dir = -1

    def stop(self):
        # Stop the motor by setting both direction pins low
        self.in1.off()  # LOW
        self.in2.off()  # LOW
        self.pwm = 0.0
        self.enable_pin.value = self.pwm
        self.dir = 0

    def change_pwm(self, pwm):
        self.pwm = min(pwm, 1.0)
        # Apply the new pwm to the PWM pin
        self.enable_pin.value = self.pwm

    def update_encoder_count(self):
        self.en_count = self.encoder.steps

    def call_encoder_interrupt(self):
        self.encoder.when_rotated = self.update_encoder_count

    def get_pos(self):
        return self.en_count