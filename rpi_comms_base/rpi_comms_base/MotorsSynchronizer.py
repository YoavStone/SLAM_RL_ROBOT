from .MotorDriver import MotorDriver


class MotorsSynchronizer:
    def __init__(self, right_pin1, right_pin2, left_pin1, left_pin2, right_pwm_pin, left_pwm_pin, pwm_frequency, right_enable_pin_a, right_enable_pin_b, left_enable_pin_a, left_enable_pin_b):
        # Initialize right motor
        # in1, in2: direction control pins
        # pwmR: enable pin for PWM speed control (ENA)
        self.R_Motor = MotorDriver(right_pin1, right_pin2, right_pwm_pin, pwm_frequency, right_enable_pin_a, right_enable_pin_b)

        # Initialize left motor
        # in3, in4: direction control pins
        # pwmL: enable pin for PWM speed control (ENB)
        self.L_Motor = MotorDriver(left_pin1, left_pin2, left_pwm_pin, pwm_frequency, left_enable_pin_a, left_enable_pin_b)

    def go_forward(self):
        self.R_Motor.forward()
        self.L_Motor.backwards()

    def go_backwards(self):
        self.R_Motor.backwards()
        self.L_Motor.forward()

    def stop(self):
        self.R_Motor.stop()
        self.L_Motor.stop()

    def turn_right(self):
        self.R_Motor.backwards()
        self.L_Motor.backwards()

    def turn_left(self):
        self.R_Motor.forward()
        self.L_Motor.forward()

    def set_pwm(self, right_pwm, left_pwm):
        right_pwm = min(abs(right_pwm), 1.0)
        left_pwm = min(abs(left_pwm), 1.0)
        self.R_Motor.change_pwm(right_pwm)
        self.L_Motor.change_pwm(left_pwm)

    def call_encoder_interrupt(self):
        self.R_Motor.call_encoder_interrupt()
        self.L_Motor.call_encoder_interrupt()

    def get_motors_pos(self):
        return self.R_Motor.get_pos(), -self.L_Motor.get_pos()