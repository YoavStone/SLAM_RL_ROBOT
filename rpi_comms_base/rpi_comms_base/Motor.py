from gpiozero import DigitalOutputDevice, PWMOutputDevice, RotaryEncoder


class Motor:
    def __init__(self, input1, input2, pwm, FREQ, ena, enb):
        # Set up direction control pins
        self.in1 = DigitalOutputDevice(input1)
        self.in2 = DigitalOutputDevice(input2)

        # Set up PWM speed control pin (ENA/ENB on the L298N)
        self.enable_pin = PWMOutputDevice(pwm, frequency=FREQ)

        # Set up encoder pins as input devices
        self.encoder = RotaryEncoder(ena, enb, max_steps=0)

        # Store pin numbers for reference
        self.pwm = pwm
        self.FREQ = FREQ

        # Initialize state variables
        self.speed = 0.75  # 0-1 scale
        self.en_count = 0
        self.dir = 0

        # Set initial speed
        self.enable_pin.value = self.speed

    def forward(self):
        print("forward")
        # Set direction pins
        self.in1.on()  # HIGH
        self.in2.off()  # LOW
        # Ensure speed is applied
        self.enable_pin.value = self.speed
        self.dir = 1

    def backwards(self):
        print("backward")
        # Set direction pins
        self.in1.off()  # LOW
        self.in2.on()  # HIGH
        # Ensure speed is applied
        self.enable_pin.value = self.speed
        self.dir = -1

    def stop(self):
        print("stop")
        # Stop the motor by setting both direction pins low
        self.in1.off()  # LOW
        self.in2.off()  # LOW
        # Could also stop by turning off the enable pin
        # self.enable_pin.off()
        self.dir = 0

    def change_speed(self, speed):
        self.speed = min(speed, 1.0)

        # Apply the new speed to the PWM pin
        self.enable_pin.value = self.speed
        print('speed: ', speed)

    def update_en_count(self):
        self.en_count = self.encoder.steps
        print(f"Encoder Count: {self.en_count}")

    def call_encoder_int(self):
        self.encoder.when_rotated = self.update_en_count

    def get_pos(self):
        return self.en_count