import serial
import time
from .ArduinoMotorDriver import ArduinoMotorDriver


class ArduinoMotorsSynchronizer:
    """
    Class to control two motors via Arduino.
    Sends commands to Arduino and receives encoder data via serial.
    """

    def __init__(self, port='/dev/ttyACM0', baudrate=115200, timeout=1.0):
        """
        Initialize Arduino motors synchronizer.

        Args:
            port: Serial port where Arduino is connected
            baudrate: Serial communication speed
            timeout: Serial read timeout
        """
        # Initialize motors
        self.R_Motor = ArduinoMotorDriver()
        self.L_Motor = ArduinoMotorDriver()

        # Connect to Arduino
        try:
            self.arduino = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # Wait for Arduino to reset
            self.arduino.flush()
            print(f"Connected to Arduino on {port}")

            # Reset encoders at startup
            self.arduino.write(b"R\n")
            response = self.arduino.readline().decode('utf-8').strip()
            print(f"Reset encoders: {response}")

        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            self.arduino = None

    def go_forward(self):
        """Set both motors to move the robot forward."""
        self.R_Motor.backwards()
        self.L_Motor.forward()
        self._send_motor_command()

    def go_backwards(self):
        """Set both motors to move the robot backward."""
        self.R_Motor.forward()
        self.L_Motor.backwards()
        self._send_motor_command()

    def stop(self):
        """Stop both motors."""
        self.R_Motor.stop()
        self.L_Motor.stop()
        self._send_motor_command()

    def turn_right(self):
        """Set motors to turn the robot right."""
        self.R_Motor.forward()
        self.L_Motor.forward()
        self._send_motor_command()

    def turn_left(self):
        """Set motors to turn the robot left."""
        self.R_Motor.backwards()
        self.L_Motor.backwards()
        self._send_motor_command()

    def set_pwm(self, right_pwm, left_pwm):
        """Set PWM values for both motors."""
        right_pwm = min(abs(right_pwm), 1.0)
        left_pwm = min(abs(left_pwm), 1.0)
        self.R_Motor.change_pwm(right_pwm)
        self.L_Motor.change_pwm(left_pwm)
        self._send_motor_command()

    def _send_motor_command(self):
        """Send current motor states to Arduino."""
        if self.arduino:
            cmd = f"M,{self.R_Motor.dir},{self.L_Motor.dir},{self.R_Motor.pwm},{self.L_Motor.pwm}\n"
            self.arduino.write(cmd.encode())
            self.arduino.flush()

            # Read acknowledgement
            response = self.arduino.readline().decode('utf-8').strip()
            if response != "OK":
                print(f"Warning: Unexpected response: {response}")

    def call_encoder_interrupt(self):
        """No-op since Arduino handles interrupts."""
        pass

    def get_motors_pos(self):
        """
        Request encoder counts from Arduino.

        Returns:
            tuple: (right_position, left_position) as encoder counts
        """
        if not self.arduino:
            return 0, 0

        try:
            # Clear input buffer
            self.arduino.reset_input_buffer()

            # Send get encoders command
            self.arduino.write(b"G\n")
            self.arduino.flush()

            # Read response
            response = self.arduino.readline().decode('utf-8').strip()

            if response.startswith("E,"):
                parts = response.split(",")
                if len(parts) == 3:
                    right_count = int(parts[1])
                    left_count = int(parts[2])

                    # Update motor encoder counts
                    self.R_Motor.update_encoder_count(right_count)
                    self.L_Motor.update_encoder_count(left_count)

                    return right_count, left_count
        except Exception as e:
            print(f"Error getting encoder counts: {e}")

        # If error or no Arduino, return current values
        return self.R_Motor.get_pos(), self.L_Motor.get_pos()

    def reset(self):
        # Reset encoders at startup
        self.arduino.write(b"R\n")
        self.R_Motor.reset()
        self.L_Motor.reset()

    def close(self):
        """Close the Arduino connection and stop motors."""
        if self.arduino:
            self.stop()
            self.arduino.close()
            print("Arduino connection closed")