# Weighted Multiple kernel ridge regression (WMKRR)

**WMKRR** is a weighted multiple kernel extension of kernel ridge regression (KRR) that integrates genotype data and gene expression data through a multi-kernel learning (MKL) strategy for genomic prediction. 

## Tutorial and Examples

We implemented WMKRR in Python. Dependencies: python > 3.6.

We provided example code and toy datasets to illustrate how to use WMKRR for hyperparameter optimization and genomic prediction. Please check WMKRR.py to see how to run WMKRR on the toy example we provided in the example_data directory. 

### Prepare files

The prepare files need to be placed in the `example_data` folder and include the following six files：

1、`X1.txt`: The genotype file for the training set individuals. each column represents a SNP marker (encoded as 0, 1, or 2). 

2、`X2.txt`: The gene expression file for the training set individuals. Each column represents a gene.

3、`y.txt`: The phenotype file for the training set individuals. 

4、`X1_test.txt`: The genotype file for the test set individuals. Each column represents a SNP marker (encoded as 0, 1, or 2). 

5、`X2_test.txt`: The gene expression file for the test set individuals. Each column represents a gene.

6、`val_id`: The individual number file for the test set individuals.

### Running command

Before running the program, the users needs to install the required packages (gc, numpy, skopt, scikit-learn, scipy, etc.). Then, place the software and the `example_data` folder in the same directory. Enter the current directory and run the program by typing the command `python WMKRR.py`. For example:

```sh
cd path/to/your/directory
python WMKRR.py
```

It should be noted that the `WMKRR.py` script includes the complete hyperparameter optimization, model fitting, and prediction steps. If we have already determined the hyperparameters of a dataset and want to directly use the known hyperparameters to fit the model and make predictions, we can directly run the `WMKRR_known_params.py` script, but we need to modify the value of the hyperparameters on line 61 of the script. (For example, in cross-validation, we first determine the hyperparameters of the dataset using WMKRR.py in several folds, and then directly use `WMKRR_known_params.py` to make predictions in all cross-validation groups). The command to run `WMKRR_known_params.py` is as follows:

```sh
cd path/to/your/directory
python WMKRR_known_params.py
```


### output files

The output files will be stored in the `results` folder and include `best_params.txt` and `valID_pred.txt`.

1、`best_params.txt`: The optimal hyperparameters determined by the Bayesian optimization algorithm.

2、`valID_pred.txt`: The predicted values for the test set individuals obtained by fitting the model using the optimal hyperparameters. The file contains two columns: individual IDs and their corresponding predicted values.



## Other Notes on the Software

- For the training set data, please make sure that the order of individuals (rows) is consistent across the genotype file (`X1.txt`), the gene expression file (`X2.txt`), and the phenotype file (`y.txt`).
- For the test set data, please make sure that the order of individuals (rows) in the genotype file (`X1_test.txt`) and the gene expression file (`X2_test.txt`) exactly matches the individual number file (`val_id`).



## QUESTIONS AND FEEDBACK

For questions or concerns with WMKRR software, please contact xwangchnm@163.com.

We welcome and appreciate any feedback you may have with our software and/or instructions.
