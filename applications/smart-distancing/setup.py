import os
import setuptools

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(THIS_DIR, 'README.md')) as readme:
    long_description = readme.read()

setuptools.setup(
    name='jetstreamer',
    version='0.1.0',  # TODO(mdegans): single source of truth for this
    description='Smart Distancing App',
    long_description=long_description,
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
    install_requires=[
        'flask',
        'numpy',
    ],
    packages=['smart_distancing'],
    entry_points={
        'console_scripts': ['smart-distancing=smart_distancing.__main__:cli_main'],
    },
    author='neuralet team',
    project_urls={
        'Bug Reports': 'https://github.com/neuralet/neuralet/issues',
        'Source': 'https://github.com/neuralet/neuralet/tree/master/applications/smart-distancing',
    },
)
