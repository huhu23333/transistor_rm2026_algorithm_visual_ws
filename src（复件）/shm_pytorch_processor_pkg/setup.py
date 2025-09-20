from setuptools import find_packages, setup

package_name = 'shm_pytorch_processor_pkg'

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
    maintainer='huhu23333',
    maintainer_email='925393268@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'shm_pytorch_processor_node = shm_pytorch_processor_pkg.shm_pytorch_processor_node:main'
        ],
    },
)
