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
         'share', package_name, 'sim_control/'),
         glob(os.path.join('sim_control/*'))),

        (os.path.join(
            'share', package_name, 'DQLRobotSLAM/'),
         glob(os.path.join('DQLRobotSLAM/*'))),

        (os.path.join(
            'share', package_name, 'CartPoleExample/'),
         glob(os.path.join('CartPoleExample/*'))),

        (os.path.join('share', package_name, 'worlds'),
         glob(os.path.join('worlds/*'))),
        # All subfolders of models (e.g., models/room_walls/*)
        *[
            (os.path.join('share', package_name, 'models', os.path.basename(d)),
             glob(os.path.join(d, '*')))
            for d in glob('models/*') if os.path.isdir(d)
        ],

    ],
    install_requires=['setuptools', 'torch', 'torchvision', 'torchaudio', 'gymnasium'],
    zip_safe=True,
    maintainer='yoav-stone',
    maintainer_email='yoav.stone@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'asymmetric_velocity_controller = sim_control.asymmetric_velocity_controller:main',
            'episode_monitor = sim_control.episode_monitor:main',

            'control_motors = RL_robot.command_robot:main',

            'follow_wall = RL_robot.follow_left_wall:main',

            'cart_pole_dqn_agent = CartPoleExample.run_dqn_agentTRY:main',
            'slam_robot_dqn_agent = DQLRobotSLAM.run_dqn_agent:main'
        ],
    },
)
