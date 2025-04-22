from pathlib import Path
import subprocess

from platforms import platform


class Backend:
    genToApex = False

    @classmethod
    def default_name(cls, app):
        return f'{app}-{cls.short_name}'

    @classmethod
    def default_code_dir(cls, machine, app):
        if cls.genToApex:
            return Path('../../apex/src') / app.group / app.name
        else:
            return Path('..') / 'generated' / machine / app.group / app.name

    @classmethod
    def default_code_file(cls, machine, app):
        return f'{cls.default_name(app.name)}.{cls.file_extension}'

    @classmethod
    def default_compile(cls, machine, app):
        compiler, flags, libs = platform(machine, cls.name)

        return [compiler, *([] if flags is None else flags),
                '-o', cls.default_bin_dir(machine, app) / cls.default_bin_file(machine, app),
                cls.default_code_dir(machine, app) / cls.default_code_file(machine, app),
                *([] if libs is None else libs)]

    @classmethod
    def default_bin_dir(cls, machine, app):
        if cls.genToApex:
            return Path('../../apex/build') / app.group / app.name
        else:
            return Path('..') / 'build' / machine / app.group / app.name

    @classmethod
    def default_bin_file(cls, machine, app):
        return f'{cls.default_name(app.name)}'

    @classmethod
    def default_measurement_dir(cls, machine, app):
        return Path('..') / 'measurements' / app.group / app.name

    @classmethod
    def default_measurement_file(cls, machine, app):
        return f'{machine}.csv'

    @classmethod
    def print_code_file(cls, machine, app, code, format=True):
        output_folder = cls.default_code_dir(machine, app)
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        output_file = cls.default_code_dir(machine, app) / cls.default_code_file(machine, app)
        print(f'    Writing to {output_file}')

        with open(output_file, 'w+') as f:
            print(code, file=f)

        if format:
            cls.format_code_file(machine, app)

    @classmethod
    def format_code_file(cls, machine, app):
        output_file = cls.default_code_dir(machine, app) / cls.default_code_file(machine, app)
        print(f'    Formatting {output_file}')

        subprocess.check_call(['clang-format', '-i', '-style=LLVM', '-style={ColumnLimit: 0, IndentWidth: 4, MaxEmptyLinesToKeep: 2}', output_file])

    @classmethod
    def compile_bin(cls, machine, app):
        output_folder = cls.default_bin_dir(machine, app)
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        subprocess.check_call(cls.default_compile(machine, app))

    @classmethod
    def exec_bin(cls, machine, app):
        bin_file = cls.default_bin_dir(machine, app) / cls.default_bin_file(machine, app)
        subprocess.check_call([bin_file] + [str(p) for p in app.default_parameters])
