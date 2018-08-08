from distutils.core import setup

setup(
    name='xaq_Lab_Task_Celii',
    version='0.1dev',
    author='Brendan Celii',
    author_email='celiibrendan@gmail.com',
    packages=['sfa'],
    install_requires=['torch','numpy','math','matplotlib'], #external packages as dependencies
    url = 'https://github.com/sheriferson/simplestatistics',
    download_url = 'https://github.com/sheriferson/simplestatistics/tarball/0.2.5',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read()
)