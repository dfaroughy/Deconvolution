
import setuptools

setuptools.setup(name="deconvolution",
                version=1.0,
                url="git@github.com:dfaroughy/Deconvolution.git",
                packages=setuptools.find_packages("src"),
                package_dir={"": "src"}
                )