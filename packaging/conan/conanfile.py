from conan import ConanFile
from conan.tools.files import copy, get
from conan.tools.gnu import Make
import os


required_conan_version = ">=2.0"


class TinycmlConan(ConanFile):
    name = "tinycml"
    version = "0.1.0"
    description = (
        "Tiny C Machine Learning Library — "
        "a lightweight C11 ML library with no dependencies"
    )
    license = "MIT"
    url = "https://github.com/username/tinycml"
    homepage = "https://github.com/username/tinycml"
    topics = ("machine-learning", "c11", "ml", "library")
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
    }

    # No dependencies — tinycml is self-contained
    def requirements(self):
        pass

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")
        self.settings.rm_safe("compiler.libcxx")
        self.settings.rm_safe("compiler.cppstd")

    def source(self):
        # TODO: Update repo and tag when publishing
        get(self, "https://github.com/username/tinycml/archive/refs/tags/v0.1.0.tar.gz",
            strip_root=True)

    def build(self):
        make = Make(self)
        # Build both static and shared; the Makefile handles both
        targets = ["static", "shared"]
        for target in targets:
            make.target = target
            make.run()

    def package(self):
        # Headers
        copy(
            self, "*.h",
            dst=os.path.join(self.package_folder, "include", "tinycml"),
            src=os.path.join(self.source_folder, "include"),
        )

        # Static library
        copy(
            self, "libtinycml.a",
            dst=os.path.join(self.package_folder, "lib"),
            src=os.path.join(self.source_folder, "build", "lib"),
            keep_path=False,
        )

        # Shared library
        copy(
            self, "libtinycml.so*",
            dst=os.path.join(self.package_folder, "lib"),
            src=os.path.join(self.source_folder, "build", "lib"),
            keep_path=False,
        )

        # License
        copy(
            self, "LICENSE",
            dst=os.path.join(self.package_folder, "licenses"),
            src=self.source_folder,
            keep_path=False,
        )

        # pkg-config
        copy(
            self, "tinycml.pc",
            dst=os.path.join(self.package_folder, "lib", "pkgconfig"),
            src=os.path.join(self.source_folder, "build"),
            keep_path=False,
        )

    def package_info(self):
        self.cpp_info.libs = ["tinycml"]
        self.cpp_info.includedirs = ["include/tinycml"]
        self.cpp_info.set_property("cmake_target_name", "tinycml::tinycml")
        self.cpp_info.set_property("pkg_config_name", "tinycml")
        if self.settings.os in ("Linux", "FreeBSD"):
            self.cpp_info.system_libs = ["m"]
