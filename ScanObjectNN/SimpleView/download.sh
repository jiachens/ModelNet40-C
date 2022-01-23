#!/bin/bash

wgetgdrive(){
  # $1 = file ID
  # $2 = file name

  URL="https://docs.google.com/uc?export=download&id=$1"

  wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -f ./cookies.txt
}

key="$1"
case $key in
    pretrained)
        wgetgdrive 13azeEzByxIw-Q_1A4EzlVtUuTHUOWCi6 pretrained.tar
        tar -xvf pretrained.tar
        ;;
    modelnet40)
        wget --no-check-certificate https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip

        unzip modelnet40_ply_hdf5_2048.zip
        
        mkdir ../data
        mv modelnet40_ply_hdf5_2048 ../data
        ;;
    *)
        echo "unknow argument $1" # unknown argument
        ;;
esac
