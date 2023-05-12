# Ontology Learning from Weak Labels

## Data

`data/`: `*.npy` files with audio features and two level ontology class indices

`metadata/`: contains mapping between label indices and label names, sources of data points (for audioset, this would be YouTube video ID, start/end time, and label)

---

## Code

### Base line Siamese Network

`*.py` and `*.ipynb` files in the root folder are for the baseline model.

`models.py`: Model architectures

`loss.py`: Loss functions

`baseline.ipynb`: Baseline Siamese Network from paper 'Sound event classification using ontology-based neural networks'

`preprocess.py`: Collects all Audioset data from TFRecord files and ontology information into dictionary

`dataloaders.py`: Dataloaders generating three types of pairs of inputs for Siamese net

`extract_audioset_features.ipynb`: Transfer pre-extracted features stored in TFRecord files to `numpy.ndarray`.

`evaluate.py`: Functions evaluating models given evaluation dataset

`plots.m`: Matlab script graphing results stored to `results/`

### Graph Convolutional Network

`audioset_wo_siamese_w_gcn/` contains all the code for the Graph Convolutional Network without ontology layer mentioned in the report.


---

## Additional Experiments

There is code for experiments that has not been reflected in the final report due to bad results or extremely long training time, but is kept for reference. They are in folder `msos`, `yamnet`, `audioset_strong_label`. 

`msos/` contains code preprocessing [Making Sense of Sounds Data](https://cvssp.org/projects/making_sense_of_sounds/site/challenge/) to fit our base line model.

`yamnet/` contains code preprocessing Audioset data with yamnet embeddings.

`audioset_strong_label` contains preprocessing and modified model for training on Audioset with strong labels.

`audioset_wo_ontology` contains all the code for the Siamese network without ontology layer model mentioned in the report.
