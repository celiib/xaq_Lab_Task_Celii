from setuptools import setup

setup(
    name='xaq_Lab_Task_Celii',
    version='0.1dev',
    author='Brendan Celii',
    author_email='celiibrendan@gmail.com',
    packages=['sfa'],
    install_requires=['torch','numpy','matplotlib'], #external packages as dependencies
    setup_requires=['torch','numpy','matplotlib'],
    url = 'https://github.com/celiib/xaq_Lab_Task_Celii',
    download_url = 'https://github.com/celiib/xaq_Lab_Task_Celii.git',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read()
)