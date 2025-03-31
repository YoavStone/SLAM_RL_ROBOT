from . import Motor


class L298nDriver:
    def __init__(self, in1, in2, in3, in4, pwmR, pwmL, FREQ, ena1, enb1, ena2, enb2):
        # Initialize right motor
        # in1, in2: direction control pins
        # pwmR: enable pin for PWM speed control (ENA)
        self.R_Motor = Motor.Motor(in1, in2, pwmR, FREQ, ena1, enb1)

        # Initialize left motor
        # in3, in4: direction control pins
        # pwmL: enable pin for PWM speed control (ENB)
        self.L_Motor = Motor.Motor(in3, in4, pwmL, FREQ, ena2, enb2)

    def go_forward(self):
        self.R_Motor.forward()
        self.L_Motor.forward()

    def go_backwards(self):
        self.R_Motor.backwards()
        self.L_Motor.backwards()

    def stop(self):
        self.R_Motor.stop()
        self.L_Motor.stop()

    def turn_right(self):
        self.R_Motor.backwards()
        self.L_Motor.forward()

    def turn_left(self):
        self.R_Motor.forward()
        self.L_Motor.backwards()

    def set_pwm(self, pwmR, pwmL):
        pwmR = min(pwmR, 1.0)
        pwmL = min(pwmL, 1.0)
        self.R_Motor.change_pwm(pwmR)
        self.L_Motor.change_pwm(pwmL)

    def call_encoder_interrupt(self):
        self.R_Motor.call_encoder_int()
        self.L_Motor.call_encoder_int()

    def get_motors_pos(self):
        return self.R_Motor.get_pos(), self.L_Motor.get_pos()