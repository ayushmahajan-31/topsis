from setuptools import setup, find_packages
from setuptools import setup, find_packages

# Read the README file with UTF-8 encoding
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='topsis102217164Ayush',
    version='0.1.0',
    description='A basic TOPSIS package for decision analysis.',
    author='Ayush',
    author_email='ayushmahajan3997@gmail.com',
    url='https://github.com/ayushmahajan-31/Topsis-package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
