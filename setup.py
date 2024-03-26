from setuptools import setup

setup(
    name='xsbert',
    version='0.1.0',    
    description='explainable sentence transformers',
    author='Lucas Moeller',
    author_email='lucasmoeller@me.com',
    license='BSD 2-clause',
    packages=['xsbert'],
    install_requires=[
        'sentence-transformers',
        'matplotlib',
        'wget'
        ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)