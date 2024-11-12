from backend.base import Base
from backend.util_header import UtilHeader

from platforms import platform

from util import *


class Makefile(Base):
    name = 'Makefile'
    short_name = 'make'

    file_extension = None

    @classmethod
    def default_code_file(cls, _):
        return f'Makefile'

    def generate(machine, app, backends):
        util_header = UtilHeader

        targets = f' \\{newline}'.join(f'\t{backend.default_bin_file(app)}' for backend in backends)

        build_rules = []
        for backend in backends:
            compiler, flags, libs = platform(machine, backend.name)  # TODO: specific util headers for cuda, hip, sycl
            build_rules.append(f'''\
$(BUILD_DIR)/{backend.default_bin_file(app)}: {backend.default_code_file(app)} {util_header.default_code_file(app)} ../../util.h
\t{compiler} {"" if flags is None else " ".join(flags)} -o $(BUILD_DIR)/{backend.default_bin_file(app)} {backend.default_code_file(app)} {"" if libs is None else " ".join(libs)}'''.strip())
        build_rules = f'{newline}{newline}'.join(build_rules)

        aliases = f'{newline}{newline}'.join(
            f'''\
.PHONY: {backend.default_bin_file(app)}
{backend.default_bin_file(app)}: $(BUILD_DIR)/{backend.default_bin_file(app)}'''
            for backend in backends)

        benchmark_targets = f'{newline}{newline}'.join(
            f'''\
\t@echo "{backend.name}:"
\t$(BUILD_DIR)/{backend.default_bin_file(app)} $(PARAMETERS)
\t@echo ""'''
            for backend in backends)

        return f'''\
# configuration

TEST_CLASS = {app.group}
TEST_CASE  = {app.name}
BUILD_DIR  = {backend.default_bin_dir(app).relative_to(backend.default_code_dir(app), walk_up=True)}


# default parameters

PARAMETERS = {" ".join(str(p) for p in app.default_parameters)}


# all

targets = \\
{targets}

.PHONY: all
all: mk-target-dir $(targets)

mk-target-dir:
	mkdir -p $(BUILD_DIR)


# build rules

{build_rules}


# aliases without build directory

{aliases}


# automated benchmark target

.PHONY: bench
bench: all
{benchmark_targets}


# clean target

.PHONY: clean
clean:
\trm $(targets)'''
