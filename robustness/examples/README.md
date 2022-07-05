# Examples

## About
Here you find a few examples on how to use the tool. Initially implemented in **PyTorch**.

## On Generating Adversarial Attacks
You can use `create_adv_examples.py` to play a bit around with adversarial attacks.
Note that all example scripts work with the [IFN_Dataloader](https://github.com/ifnspaml/IFN_Dataloader) and they expect that the [Cityscapes](https://www.cityscapes-dataset.com/) dataset is downloaded.
Also, make sure that you change the lines, where the model is loaded as you will need a (trained) model to test the scripts.

## On Creating Common Image Corruptions
You can use `create_corruptions.py` to play a bit around with common image corruptions.
Note that all example scripts work with the [IFN_Dataloader](https://github.com/ifnspaml/IFN_Dataloader) and they expect that the [Cityscapes](https://www.cityscapes-dataset.com/) dataset is downloaded.
Also, make sure that you change the lines, where the model is loaded as you will need a (trained) model to test the scripts.

## On Creating Cityscapes-C
Depending on what you are intending to do, it might be reasonable to preprocess your data.
As the image corruptions and severity levels in Cityscapes-C are well defined, you can just preprocess the data and create your own Cityscapes-C dataset.
Exactly this is the intention behind using `create_cityscapes_c.py`.
Just create Cityscapes-C and save some time when performing an evaluation.
