from setuptools import setup, Extension, find_packages

repmetric_ext = Extension(
    "repmetric._cpp",
    sources=["src/repmetric/cped.cpp", "src/repmetric/levd.cpp"],
    extra_compile_args=["-O3", "-std=c++17"],
)

setup(
    name="repmetric",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[repmetric_ext],
    zip_safe=False,
)
