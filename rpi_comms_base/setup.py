from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'rpi_comms_base'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join(
         'share', package_name, 'rpi_motor_control'),
         glob(os.path.join('rpi_motor_control/*'))),

        (os.path.join(
            'share', package_name, 'rpi_comms_base'),
         glob(os.path.join('rpi_comms_base/*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoavstone',
    maintainer_email='yoav.stone@gmail.com',
    description='rpi talks to base computer',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'script = rpi_comms_base.script_try:main',
            'control_motors = rpi_comms_base.run_robot_control_node:main',
            'TFBroadCasterRobot = rpi_comms_base.TFBroadCasterRobot:main',
        ],
    },
)
