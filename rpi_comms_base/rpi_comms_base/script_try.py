from .MotorsSynchronizer import MotorsSynchronizer


def is_number_between_0_and_1(text):
    try:
        number = float(text)
        return 0 <= number <= 1
    except ValueError:
        return False


def main():
    # With gpiozero we don't need to open a chip

    # Define GPIO pins
    in1 = 17
    in2 = 27
    pwmR = 22  # Not directly used with gpiozero but kept for compatibility
    ena1 = 19
    enb1 = 26

    in3 = 23
    in4 = 24
    pwmL = 25  # Not directly used with gpiozero but kept for compatibility
    ena2 = 13
    enb2 = 6

    FREQ = 1000  # 1000 HZ - Note: gpiozero handles PWM frequency internally

    # Create driver without chip handle
    driver = MotorsSynchronizer(in1, in2, in3, in4, pwmR, pwmL, FREQ, ena1, enb1, ena2, enb2)
    driver.call_encoder_interrupt()

    print("\n")
    print("The default speed of motor is 75%")
    print("q-stop w-forward s-backward d-right a-left num_0-1-speed p-motors_pos e-exit")
    print("\n")

    while True:
        x = input()

        if x == 'q':
            driver.stop()

        elif x == 'w':
            driver.go_forward()

        elif x == 's':
            driver.go_backwards()

        elif x == 'd':
            driver.turn_right()

        elif x == 'a':
            driver.turn_left()

        elif is_number_between_0_and_1(x):
            x = float(x)
            x = round(x, 2)  # two digits after decimal point
            driver.set_pwm(x, x)

        elif x == 'p':
            print(driver.get_motors_pos())

        elif x == 'e':
            driver.stop()
            # No need for explicit cleanup with gpiozero - it handles this automatically
            print("Exiting program")
            break

        else:
            print("<<<  wrong data  >>>")
            print("please enter the defined data to continue.....")


if __name__ == "__main__":
    main()