from setuptools import setup, find_packages

setup(
    name='plntxps',
    version='0.1.0',    
    description='XPS analysis tools',
    #url='https://github.com/ChristianGeci/fluorCalc',
    author='Christian Geci',
    author_email='christian.geci@maine.edu',
    license='MIT license',
    packages = find_packages(),
    install_requires=[
                      'numpy',
                      'scipy',
                      'matplotlib',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        #'License :: OSI Approved :: BSD License',  
        #'Operating System :: POSIX :: Linux',        
        #'Programming Language :: Python :: 2',
        #'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.4',
        #'Programming Language :: Python :: 3.5',
    ],
    package_data={
        "plntxps.resources": ["HandbookXPS.csv"],
    },
)
