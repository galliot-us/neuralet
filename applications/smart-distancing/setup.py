import os
import setuptools

PACKAGE_NAME = 'smart_distancing'
DESCRIPTION = 'Smart Distancing App'

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(THIS_DIR, PACKAGE_NAME)

# get some setup variables from files
with open(os.path.join(THIS_DIR, 'README.md')) as readme:
    LONG_DESCRIPTION = readme.read()
with open(os.path.join(THIS_DIR, 'requirements.in')) as requirements:
    INSTALL_REQUIRES = [l.strip() for l in requirements]
with open(os.path.join(PACKAGE_DIR, 'VERSION')) as version_file:
    VERSION = version_file.readline().strip()[:16]

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',

        'License :: OSI Approved :: Apache License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',
    install_requires=INSTALL_REQUIRES,
    packages=[
        'smart_distancing',
        'smart_distancing.core',
        'smart_distancing.detectors',
        'smart_distancing.detectors.edgetpu',
        'smart_distancing.detectors.jetson',
        'smart_distancing.detectors.x86',
        'smart_distancing.loggers',
        'smart_distancing.tools',
        'smart_distancing.ui',
        'smart_distancing.ui.utils',
        'smart_distancing.utils',
    ],
    package_data={
        'smart_distancing': [
            'VERSION',
            'data/logs',
            'data/config/*.ini',
            'data/models/jetson',
            'data/models/edgetpu',
        ],
        'smart_distancing.ui': [
            'static/*.js',
            'templates/*.html',
        ],
    },
    entry_points={
        'console_scripts': ['smart-distancing=smart_distancing.__main__:cli_main'],
    },
    author='neuralet team',
    project_urls={
        'Bug Reports': 'https://github.com/neuralet/neuralet/issues',
        'Source': 'https://github.com/neuralet/neuralet/tree/master/applications/smart-distancing',
    },
)
