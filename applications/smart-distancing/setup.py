import os
import setuptools

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(THIS_DIR, 'README.md')) as readme:
with open(os.path.join(THIS_DIR, 'requirements.in')) as requirements:
    INSTALL_REQUIRES = [l.strip() for l in requirements]
with open(os.path.join(PACKAGE_DIR, 'VERSION')) as version_file:
    VERSION = version_file.readline().strip()[:16]

setuptools.setup(
    version=VERSION,
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
    packages=['smart_distancing'],
    package_data={
        'smart_distancing': [
            'VERSION',
            'data/logs',
            'data/config/*.ini',
            'data/models/jetson',
            'data/models/edgetpu',
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
