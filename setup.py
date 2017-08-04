from setuptools import setup

setup(
    name='deepshinx',
    version='0.1',
    packages=['deepshinx'],
    scripts=['bin/deepshinx_train'],
    description='Trainer',
    # Replace with tensorflow-gpu if using GPU
    install_requires=['python_speech_features',
                      'tensorflow==1.3',
                      'scipy',
                      'pysoundfile',
                      'openfst'])
