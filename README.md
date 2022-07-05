<img align="right" src="../img/logo_IFN.svg" alt="logo_ifn"> <br/>

# TUBSRobustCheck

## About
TUBSRobustCheck is a toolbox to evaluate the robustness of semantic segmentation networks.
The robustness is evaluated in terms of robustness towards adversarial attacks ([attacks](https://github.com/ifnspaml/TUBSRobustCheck/tree/main/robustness/attacks) as well as robustness towards common corruptions ([corruptions](https://github.com/ifnspaml/TUBSRobustCheck/tree/main/robustness/corruptions)).
It is mainly developed by [Andreas Bär](https://scholar.google.de/citations?user=QMkFXXgAAAAJ&hl=de) and [Edgard Moreira Minete]() at the [Institute for Communications Technology](https://www.tu-braunschweig.de/en/ifn/institute/dept/sv) of the Technische Universität Braunschweig.


## [robustness/attacks](https://github.com/ifnspaml/TUBSRobustCheck/tree/main/robustness/attacks)
This tool implements various individual and universal adversarial attacks for semantic segmentation networks.
More information can be found [here](https://github.com/ifnspaml/TUBSRobustCheck/tree/main/robustness/attacks).


## [robustness/corruptions](https://github.com/ifnspaml/TUBSRobustCheck/tree/main/robustness/corruptions)
This tool implements common corruptions, first introduced by [(Hendrycks et al., ICLR 2019)](https://arxiv.org/pdf/1903.12261.pdf).
More information can be found [here](https://github.com/ifnspaml/TUBSRobustCheck/tree/main/robustness/corruptions).


## Installation
To install TUBSRobustCheck, the following packages are needed:
```
conda create --name robust_check python==3.7 # Python 3.8 & 3.9 throw some gcc issues
source activate robust_check
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install matplotlib
pip install opencv-python
conda install -c conda-forge wand
conda install scipy
conda install scikit-image
conda install numba
pip install git+https://github.com/ifnspaml/IFN_Dataloader.git # This is only necessary if one wants to use the dataloader from the examples
```

To install via environment.txt, follow the following steps:
```
conda env create -f environment.txt
pip install git+https://github.com/ifnspaml/IFN_Dataloader.git
```
For reference: The environment.txt were created by exporting the environment via `conda list -e > environment.txt` on a Linux cluster.
Thus, they only work with Linux.

For more information about the dataloader please visit the [**IfN dataloader**](https://github.com/ifnspaml/IFN_Dataloader) repository.


## Installation as a package
To install TUBSRobustCheck as a package, run the following command inside your environment:

```
pip install -e .
```


## Shoutout
We would like to thank all contributors of
https://github.com/Harry24k/adversarial-attacks-pytorch,
https://github.com/bethgelab/imagecorruptions/pull/18,
https://github.com/Trusted-AI/adversarial-robustness-toolbox,
https://github.com/ej0cl6/pytorch-adversarial-examples,
https://github.com/jsikyoon/nips17_adv_attack,
https://github.com/Harry24k/AEPW-pytorch,
and https://github.com/hendrycks/robustness for their awesome work.
We used these GitHub repositories as guidelines and inspiration to create our own code base serving our needs.
Please check out their contributions as well!


## License
MIT Licence
