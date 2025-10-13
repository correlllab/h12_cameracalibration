from setuptools import find_packages, setup

package_name = 'h12_cameracalibration'

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
    maintainer='max',
    maintainer_email='maxlconway@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'collect_handineye_calib = h12_cameracalibration.collect_handineye_calibration_data:main',
            'collect_handtoeye_calib = h12_cameracalibration.collect_handtoeye_calibration_data:main',
        ],
    },
)
