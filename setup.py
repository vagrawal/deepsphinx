from setuptools import setup

setup(
    name='deepshinx',
    version='0.1',
    packages=['deepsphinx'],
    scripts=['bin/deepsphinx_train'],
    description='Trainer',
    # Replace with tensorflow-gpu if using GPU
    install_requires=['python_speech_features',
                      'tensorflow==1.3.0',
                      'scipy',
                      'pysoundfile',
                      'openfst'])
