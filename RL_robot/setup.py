from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'RL_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join(
         'share', package_name, 'launch'),
         glob(os.path.join('launch/*'))),

        (os.path.join(
         'share', package_name, 'model'),
         glob(os.path.join('model/*'))),
        
        (os.path.join(
         'share', package_name, 'parameters'),
         glob(os.path.join('parameters/*'))),

        (os.path.join(
         'share', package_name, 'worlds'),
         glob(os.path.join('worlds/*'))),

        (os.path.join(
         'share', package_name, 'RL_robot/'),
         glob(os.path.join('RL_robot/*'))),

        (os.path.join(
         'share', package_name, 'rviz/'),
         glob(os.path.join('rviz/*'))),

        (os.path.join(
         'share', package_name, 'robot_controller/'),
         glob(os.path.join('robot_controller/*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoav-stone',
    maintainer_email='yoav.stone@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'follow_wall = RL_robot.follow_left_wall:main',
            'control_motors = RL_robot.control_motors:main',
        ],
    },
)
