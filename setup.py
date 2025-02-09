from setuptools import setup

setup(
   name='tensor_layers_CoMERA',
   version='1.0',
   description='CoMERA module',
   author='foo',
   author_email='foomail@foo.example',
   packages=['tensor_layers_CoMERA'],  #same as name
   install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
)
