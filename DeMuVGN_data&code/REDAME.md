# DeMuVGN

This repo provides the code for reproducing the experiments in DeMuVGN: Effective Software Defect Prediction Model by Learning Multi-view Program Dependency via Graph Neural Networks. Specifically, we propose a Multi-view Program Dependency Graph (MPDG) by combining data dependency, call dependency, and developer dependency information, and utilize the enhanced Bidirectional Gated Graph Neural Network to realize automatic learning of software features and defect prediction. 

## Datasets

We build dependency graphs datasets and feature datasets from six open-source software projects across 16 versions to verify the effectiveness of DeMuVGN in the within-project and cross-project contexts. 

## Dependency

This code is written in python 3. You will need to install the required packages in order to run the code. We recommend using conda virtual environment for the package management. Install the package requirements with `pip install -r requirements.txt`.

## Start

First, you should download the dataset from our link.

You can reproduce the results of within-project link recovery by running the link or reproduce the results of cross-project link recovery by running the link.

Please follow the instructions to complete the reproduction: [Within-project defect prediction](within-project/README.md), [Cross-project defect prediction](cross-project/README.md).

## Result

You can find the results of DeMuVGN and baselines on the within-project defect prediction task and the cross-project defect prediction task under the results folder.