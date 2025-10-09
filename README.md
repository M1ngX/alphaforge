# alphaforge
This is a simplified and unofficial implementation of AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors.

The framework is completely decoupled with qlib and the original implementation of Alphagen, and you are allowed to use your own data source as well as your own label. The key parameters, such as the max length of the alpha formular and the look back window of the operators, can also be easliy adjusted. A simple demo is provided in the jupyter notebook.

Please note that the mined factors can be furthur combined using machine learning models (as I personally recommend), so this framework does not include the "dynamic combination" part of the original implementation.

Reference:
1. The original paper of AlphaForge
2. https://github.com/DulyHao/AlphaForge
3. https://github.com/nshen7/alpha-gfn
