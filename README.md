# DrumClassifier
## A simple and lightweight drum classification model
I feel like this is getting a bit repetitive but... DrumClassifier is a drum classification model, which (currently) classifies audio samples as _kick_, _snare_, _hihat_, _toms_ or _cymbals_. The model works on audio files as long as 4 seconds - an arbitrary cut that seemed reasonable enough to me for single drum samples.

## Usage
First of all, **get the model weights from the Releases tab!**

For convenience, a wrapper for the model (_classifier.py_) is included. Simply create a _DrumClassifier_ object, passing it the appropriate _weights_path_ and _cfg_path_, and classify data using either _classify()_ or _classify_from_path()_ depending on your needs.

If you wish to quickly test the model with your own sample:
```
python tester.py -i audio_sample_path -w weights_path [-c cfg_path]
```

## Requirements
```
torch
torchaudio
einops
omegaconf
```

## Model details
The model features a 3-layer-deep 2D convolutional block, followed by 2 RNNs, one modelling channel/frequency, one modelling time sequences, for dimensionality reduction; finally, as a natural last processing step, the network features 3 linear layers for classification.

## Dataset details
The network was trained on a private dataset containing on average 425 samples per class, built using various sample libraries and individual drum hits from [StemGMD](https://zenodo.org/records/7882857). The dataset contained both acoustic and electronic drum samples, though in the latter case mainly realistic sounding electronic drums were chosen (e.g. no TR-808 sounds).

A test set containing 80 samples per class was built using unseen samples from the same sources, and a second, smaller one was made using completely new samples from different sources to assess the impact of data source biases and overfitting.

Results on the main test set indicate an accuracy of over 90% on all classes but _hihat_, where a respectable 86.25% was instead reached. Overfitting and biases are indeed present, but they do not seem to significantly impact results on new data on the second test set.

## Data augmentation techniques
The following data augmentation techniques were employed during training:
* LR channel swap
* Stereo image alterations
* Polarity reversal
* Pitch-shifting
* Translation in time
* Distortion
* MP3 compression
* Noise addition

## Future works
* Training with a bigger dataset, bringing also fully electronic drums into the picture
* New model architectures
* tester.py: Test entire folders
