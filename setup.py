try:
    from setuptools import setup, find_packages
    have_setuptools = True
except ImportError:
    from distutils.core import setup
    have_setuptools = False

if __name__ == '__main__':
    setup()  # see configuration in setup.cfg
