#!/bin/bash

PYTHON="/usr/local/bin/python3.10"

pushd deep_recommenders/datasets
$PYTHON movielens.py
popd
