# ΟmniΜatch

This repo includes the code used for implementing OmniMatch, as described in "OmniMatch: Joinability Discovery in Data Products".

## Repo structure

* [`src`]() Contains python source fiiles for developing OmniMatch and baselines used in the paper:
	- `training_generator.py` contains the code for generating training dataset pairs for self-supervision.
	- `featurizer.py` contains the code for computing column pairwise similarity metrics
	- `omnimatch_predictors.py` contains code for training and testing OmniMatch models, as described in the paper.
	- `rf_predictor.py` contains code for training and testing the Random Forest model baseline.
	- other source files needed for execution.
* [`config_files`]() Contains configuration files for each python script included in `src`.

## Datasets and other files

The dataset can be downloaded from this location: https://zenodo.org/records/15705578

Details:

* `data-products-matching/datasets` contains test and train datasets for both our join benchmarks.
* `data-products-matching/assets/features` contains column-pairwise similarity metrics for each measure used in the paper (in .pickle format) for both our join benchmarks and their corresponding test and train datasets.
* `data-products-matching/assets/samples` contains samples of training datasets that can be used for training, for both join benchmarks.
* `data-products-matching/assets/matches` contains all join and non-join pairs of training and test datasets for both our join benchmarks (in .pickle format).

## Running OmniMatch

1. In the absence of training data, use `src/training_generator.py` to generate training dataset pairs based on the test data. Make sure after generating the data to compute the full lists of join/non_join pairs between the generated dataset pairs in the format of [((filename1.csv, column1), (filename2.csv, column2)), etc.] and store them into two separate pickle files (like the ones we provide for our benchmarks). Parameters can be set through the `config_files/training_generator_config.ini` file.
2. Make sure that you also have full lists of all join and non join column pairs for the test datasets, again in .pickle format (as the ones we provide).
3. Run `src/featurizer.py` for each different metric to compute for all join and non join pairs in training/test datasets. For example if you want compute embedding similarity based on frequent values, make sure to set `value_embeddings: True` in the corresponding configuration file (`config_files/featurizer_config.ini`), while all other should be set to `False`.
4. Run `src/omnimatch_predictors.py` by setting appropriately the parameters in `config_files/omnimatch_predictors_config.ini`. To run Omnimatch set `model: rgcn_margin or rgcn_cross_entropy`, depending on the loss function you want to use.

### Example run
```
python src/omnimatch_predictors.py -cf config_files/omnimatch_predictors_config.ini
```

### Example config file (omnimatch\_predictors\_config.ini)

```
train_datasets_path: [root]/datasets/city_government/train_tables
train_features_path: [root]/features/city_government/train_tables/
train_node_features: [root]/features/city_government/individual_features.pickle
test_features_path: [root]/features/city_government/test_tables/
test_node_features:  [root]/features/city_government/test_tables/individual_features.pickle
results_path: [root]/results
samples_path: [root]/samples/city_government
sampled_datasets: (nothing as we will sample the training data in this run)

[PARAMETERS]
benchmark: city_government (run the city_government benchmark)
graph_construction: topk (keep topk edges per node)
model_loss: rgcn_margin (run OmniMatch with triplet margin loss)
k: 3
number_of_datasets: 2 (2 generated datasets per test dataset - 
should be generated beforehand with training_generator.py)
number_of_sources: 20 (use 20 of the test datasets to pick generated training pairs - 
should be generated beforehand with training_generator.py)
dimension: 256
epochs: 30
learning_rate: 0.001
margin: 0.5
norm: 2 (doesn't matter since we picked margin loss, would matter if we picked rgcn_cross_entropy)

[FEATURES]
jaccard_frequent: True
value_embeddings: True
value_distribution: True
jaccard_containment: True

[SAVEFILES]
write_embeddings: False - we don't want to store produced embeddings
write_results: True - we want to store results
```
