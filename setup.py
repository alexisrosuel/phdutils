import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='phd_utils',
     version='0.2',
     author="Alexis Rosuel",
     author_email="rosuelalexis1@gmail.com",
     description="phd useful methods",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/alexisrosuel/phd_utils",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
