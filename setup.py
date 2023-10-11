from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('taisti_linker/requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='taisti_linker',
    version='1.0.0',
    author='Dawid Wisniewski, Agnieszka Lawrynowicz',
    author_email='dwisniewski/alawrynowicz[guesswhat]cs.put.poznan.pl',
    description='Entity linker for TAISTI project',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/taisti/entity_linker',
    project_urls = {
        "Bug Tracker": "https://github.com/taisti/entity_linker/issues"
    },
    packages=find_packages(),
    license='MIT',
    # install_requires=required,
)
