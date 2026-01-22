from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'BoxFusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob(os.path.join('config', '*.rviz'))),     # rviz configs
        (os.path.join('share', package_name), glob(os.path.join('config', '*.xml'))),      # plotjuggler configs

        # launch files
        (os.path.join('share', package_name), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    include_package_data=True,
    zip_safe=True,
    maintainer='jorisv',
    maintainer_email='jorisv@kth.se',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'run_main = BoxFusion.main:main',
        ],
    },
)
