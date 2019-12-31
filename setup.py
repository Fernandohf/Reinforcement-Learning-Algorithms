from distutils.core import setup
setup(
  name = 'bistrain',
  packages = ['bistrain'],
  version = '0.1',
  license = 'MIT',
  description = '''Simple library of Reinforcement Learning Algorithms
                 implemented in Pytorch''',
  author = 'Fernando Henrique',
  author_email = 'nandohfernandes@gmail.com',
  url = 'https://github.com/Fernandohf/Reinforcement-Learning-Algorithms',
  download_url = 'https://github.com/Fernandohf/Reinforcement-Learning-Algorithms/archive/0.1.tar.gz',  # TODO
  keywords = ['PYTORCH', 'A2C', 'DDPG', 'REINFORCEMENT-LEARNING', 'RL'],
  install_requires=[
          'torch',
          'gym',
          'tqdm',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers/Researchers',
    'Topic :: Software Development :: Build Tools::',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
