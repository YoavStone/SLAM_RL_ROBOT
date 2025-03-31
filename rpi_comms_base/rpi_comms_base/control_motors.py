import lgpio

from . import L298nDriver


def is_number_between_0_and_1(text):
    try:
        number = float(text)
        return 0 <= number <= 1
    except ValueError:
        return False


class ControlMotors:
    def __init__(self):
        # Initialize the lgpio chip
        self.chip = lgpio.gpiochip_open(0)
        self.linearSpeedConst = 1.0
        self.turn_speed = 1.0

        in1 = 17
        in2 = 27
        pwmR = 22
        ena1 = 19
        enb1 = 26

        in3 = 23
        in4 = 24
        pwmL = 25
        ena2 = 13
        enb2 = 6

        FREQ = 1000  # 1000 HZ

        self.driver = L298nDriver.L298nDriver(in1, in2, in3, in4, pwmR, pwmL, FREQ, ena1, enb1, ena2, enb2)
        self.driver.call_encoder_interrupt()


    def convert_vel_to_cmd(self, speed, turn):

        if speed == 0 and turn == 0:
            print("conv vel STOP")
            self.driver.stop()

        if 0 < abs(speed) <= 1:
            vel = float(abs(speed))*self.linearSpeedConst
            vel = round(vel, 2)  # two digits after decimal point
            self.driver.set_pwm(vel, vel)
            print("conv vel VEL: ", vel)

        if speed > 0:
            print("conv vel FORWARD")
            self.driver.go_forward()

        elif speed < 0:
            print("conv vel BACK")
            self.driver.go_backwards()

        elif turn > 0:
            print("conv vel TURN RIGHT")
            self.driver.set_pwm(self.turn_speed, self.turn_speed)
            self.driver.turn_right()

        elif turn < 0:
            print("conv vel TURN LEFT")
            self.driver.set_pwm(self.turn_speed, self.turn_speed)
            self.driver.turn_left()
