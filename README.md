This project is not maintained.
It has been published as part of the following conference paper at IJCNN 2022:
# Anomaly Detection by Recombining Gated Unsupervised Experts
## by Jan-Philipp Schulze, Philip Sperl and Konstantin Böttinger

Anomaly detection has been considered under several extents of prior knowledge.
Unsupervised methods do not require any labelled data, whereas semi-supervised methods leverage some known anomalies.
Inspired by mixture-of-experts models and the analysis of the hidden activations of neural networks, we introduce a novel data-driven anomaly detection method called ARGUE.
Our method is not only applicable to unsupervised and semi-supervised environments, but also profits from prior knowledge of self-supervised settings.
We designed ARGUE as a combination of dedicated expert networks, which specialise on parts of the input data.
For its final decision, ARGUE fuses the distributed knowledge across the expert systems using a gated mixture-of-experts architecture.
Our evaluation motivates that prior knowledge about the normal data distribution may be as valuable as known anomalies.

### Dependencies
We used ``docker`` during the development.
You can recreate our environment by:  
``docker build -t argue ./docker/``.

Afterwards, start an interactive session while mapping the source folder in the container:  
``docker run --gpus 1 -it --rm -v ~/path/to/argue/:/app/ -v ~/.keras:/root/.keras argue``

#### Data sets
The raw data sets are stored in ``./data/``.
You need to add CovType [1], EMNIST [2], Census [3], Darknet [4], DoH [5], IDS [6], NSL-KDD [7] and URL [8] from the respective website.

For example, the URL's archive contains the file ``All.csv``.
Move it to ``./data/url/All.csv``.
The rest is automatically handled in ``./libs/DataHandler.py``, where you find more information which file is loaded.

#### Baseline methods
The baseline methods are stored in ``./baselines/``.
Whereas we implemented A3, MEx-CVAE, fAnoGAN, Deep-SAD and Deep-SVDD, you need to add DAGMM [9], GANomaly [10], DevNet [11] and REPEN [12] manually from the respective website.

### Instructions

#### Train models
For each data set, all applicable experiments are bundled in the respective ``do_*.py``.
You need to provide a random seed and whether the results should be evaluated on the "val" or "test" data, e.g. ``python ./do_mnist.py 123 val``.
Optional arguments are e.g. the training data pollution ``--p_contamination`` and the number of known anomalies ``--n_train_anomalies``.
Please note that we trained the models on a GPU, i.e. there will still be randomness while training the models.
Your models are stored in ``./models/`` if not specified otherwise using ``--model_path``.

#### Evaluate models
After training, the respective models are automatically evaluated on the given data split.
As output, a ``.metric.csv``, ``.roc.csv`` and ``.roc.png`` are given.
By default, these files are stored in ``./models/{p_contamination}_{random_seed}/``.
The first file contains the AUC & AP metrics, the latter two show the ROC curve.
The test results are the merged results of five runs:  
``python3 evaluate_results.py 110 210 310 410 510 610 710``

### Known Limitations
We sometimes had problems loading the trained models in TensorFlow's eager mode.
Please use graph mode instead.

### File Structure
```
ARGUE
│   do_*.py                     (start experiment on the respective data set)
│   evaluate_results.py         (calculate the mean over the test results)
│   README.md                   (file you're looking at)
│
└─── data                       (raw data)
│
└─── docker                     (folder for the Dockerfile)
│
└─── libs
│   └───architecture            (network architecture of the alarm and expert networks)
│   └───network                 (helper functions for the NNs)
│   │   ARGUE.py                (main library for our anomaly detection method)
│   │   Cluster.py              (automatic clustering for monolithic data sets)
│   │   DataHandler.py          (reads, splits, and manages the respective data set)
│   │   ExperimentWrapper.py    (wrapper class to generate comparable experiments)
│   │   Metrics.py              (methods to evaluate the data)
│
└─── models                     (output folder for the trained neural networks)
│
└─── baselines                  (baseline methods)
│
```

### Links
* [1] https://archive.ics.uci.edu/ml/datasets/Covertype
* [2] https://www.nist.gov/itl/products-and-services/emnist-dataset
* [3] https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)
* [4] https://www.unb.ca/cic/datasets/darknet2020.html
* [5] https://www.unb.ca/cic/datasets/dohbrw-2020.html
* [6] https://www.unb.ca/cic/datasets/ids-2018.html
* [7] https://www.unb.ca/cic/datasets/nsl.html
* [8] https://www.unb.ca/cic/datasets/url-2016.html
* [9] https://github.com/tnakae/DAGMM
* [10] https://github.com/chychen/tf2-ganomaly
* [11] https://github.com/GuansongPang/deviation-network
* [12] https://github.com/GuansongPang/deep-outlier-detection
