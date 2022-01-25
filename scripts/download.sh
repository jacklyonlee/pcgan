#! /bin/bash

root=$(pwd)

# Download ShapeNet15k dataset
mkdir data 
gdown --id 1R5Ku23QssKsAYTlchroQBt4czIPhFWLr -O data/data.zip
unzip data/data.zip -d data
rm data/data.zip

# Download trained checkpoints
mkdir checkpoints
mkdir checkpoints/trained
gdown --id 1TEiYM2o2YG-MVAe0YQrAkdMYs6qRsZ4E -O checkpoints/trained.zip
unzip checkpoints/trained.zip -d checkpoints/trained
rm checkpoints/trained.zip

cd "$root" || exit
