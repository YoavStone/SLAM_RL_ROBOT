import rclpy

from .VelocityMiddleware import VelocityMiddleware


def main(args=None):
    rclpy.init(args=args)
    controller = VelocityMiddleware()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()