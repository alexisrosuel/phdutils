# import setuptools
# with open("README.md", "r") as fh:
#     long_description = fh.read()
# setuptools.setup(
#      name='phdutils',
#      version='0.2',
#      author="Alexis Rosuel",
#      author_email="rosuelalexis1@gmail.com",
#      description="phd useful methods",
#      long_description=long_description,
#    long_description_content_type="text/markdown",
#      url="https://github.com/alexisrosuel/phdutils",
#      packages=setuptools.find_packages(),
#      classifiers=[
#          "Programming Language :: Python :: 3",
#          "License :: OSI Approved :: MIT License",
#          "Operating System :: OS Independent",
#      ],
#  )



import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setuptools.setup(
    name="phdutils",
    version="0.0.3",
    description="Useful methods",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/alexisrosuel/phdutils",
    author="Alexis Rosuel",
    author_email="office@realpython.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=setuptools.find_packages(),
    #include_package_data=True,
    #install_requires=["feedparser", "html2text"],
    #
)
