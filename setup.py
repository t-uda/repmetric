from setuptools import Extension, setup

repmetric_ext = Extension(
    "repmetric._cpp",
    sources=[
        "src/repmetric/cped.cpp",
        "src/repmetric/bicped.cpp",
        "src/repmetric/levd.cpp",
    ],
    extra_compile_args=["-O3", "-std=c++17"],
)

setup(
    ext_modules=[repmetric_ext],
)
