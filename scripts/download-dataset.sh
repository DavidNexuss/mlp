#!/bin/sh
function getnet {  
  wget $1
  unzip *.zip
  rm *.zip
}

mkdir -p assets
cd assets

getnet "https://github.com/teavanist/MNIST-JPG/raw/refs/heads/master/MNIST%20Dataset%20JPG%20format.zip"
