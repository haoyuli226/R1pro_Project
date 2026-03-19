from setuptools import find_packages, setup

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='yiyao.wang@vipl.ict.ac.cn',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_control = robot_controller.robot_control:main',
            'bc_controller = robot_controller.bc_controller:main',
            'keypoint_visualizer = robot_controller.keypoint_visualizer:main',
            'view = robot_controller.view:main',
        ],
    },
)
