from distutils.core import setup, Extension

version = "0.0.1"

setup(name="anba4",
      version=version,
      packages=["anba4", "anba4.material"],
      package_data={'anba4': ['material/material.cpp']}
      )
