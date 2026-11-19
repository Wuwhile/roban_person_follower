from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup(**generate_distutils_setup(
    packages=["roban_person_follower", "roban_person_follower.bytetrack"],
    package_dir={"": "src"},
))
