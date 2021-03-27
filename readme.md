# Adversarial Autoencoder applied to the Kunst der Fuge dataset.

This repository is an adaptation of the MusAE model created by [Valenti et al.](https://arxiv.org/abs/2001.05494). For theoretical background, please check their original paper.
For explicit samples of interpolation please check the repo this one has been forked from as well as [Valenti's YouTube channel](https://www.youtube.com/playlist?list=PLxrPCQsIK9XVVpTIun9meuPcOdWaG-aSg) and [the Google Drive directory provided by Valenti](https://drive.google.com/open?id=1fr16B2MGVAtyk3W4D3SgI2Z983lKCE2U).

## Installation/Dependencies

The code works with Python 3.6. Other versions have not been tested (yet)

Either use the dockerfile ''dockerfile_sem'' or manually install the following packages:
- Tensorflow 1.15.0 (either with or without GPU support, you're free to choose)
- Keras 2.2.4
- matplotlib
- pypianoroll 0.5.3
- pretty_midi
- sklearn
- h5py 2.10.0
