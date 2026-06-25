#!/bin/bash
set -ex

PYTHON="python3.10"

pushd deep_recommenders/datasets
$PYTHON movielens.py
popd
