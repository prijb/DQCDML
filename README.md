# DQCDML

This repository is designed to perform preprocessing, training and evaluation of a Boosted Decision Tree (BDT) for signal classification. The signature of focus is displaced dimuons in the Parking and Scouting datasets. The workflow is heavily based off the Icenet library ([repo](https://github.com/mieskolainen/icenet)).

## Structure

The workflow uses a steering yaml file located in `config/` to specify input, training and plotting variables along with the use of preprocessing and filtering routines.

Data are preprocessed and cached using the `Preprocessor` class in `utils/preprocess.py` which takes a file list, the config yaml location along with options for batching. Helper functions can be introduced into the Preprocessing routine by implementing them in `helper_funcs/` and calling in the config. 

ML methods with more complex implementations can be implemented in `mva/` and imported as an object.

## Examples

### Parking
`scripts/test_preprocess_parking.py` uses ntuples produced via [nanotron](https://github.com/prijb/nanotron/tree/Parking) which builds on the standard NanoAOD ntuplization with the addition of dimuon vertexing and trigger information. Steered by `config/vars_parking.yml`.

### Scouting
`scripts/test_preprocess_scouting.py` uses ntuples produced by the Looper in the [run3_scouting repository](https://github.com/cmstas/run3_scouting/tree/master). Dedicated functions in `config/vars_scouting.yml` with helper functions in `helper_funcs/scouting.py` used to produce trigger decision information and add the dimuon mass to secondary vertices.

