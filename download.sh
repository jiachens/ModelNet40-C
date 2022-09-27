wgetgdrive(){

  # $1 = file ID
  # $2 = file name

  URL="https://docs.google.com/uc?export=download&id=$1"

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

mkdir -p tmp
key="$1"
case $key in
	pretrained)
		wgetgdrive 1qSkMYYK1qkT4wMMeAXerSI2Q7AxWujsS tmp/pretrained.zip
		unzip -o tmp/pretrained.zip
		;;
	runs)
		mkdir -p runs 
		cd runs
		python ../gdrivedl.py https://drive.google.com/drive/folders/1UT-OfAsQ1OGSa6HSLZcK6YyJeIkaJUfF?usp=sharing 
    	cd ..
		;;
	cor_exp)
		mkdir -p cor_exp 
		cd cor_exp
		python ../gdrivedl.py https://drive.google.com/drive/folders/1iYcJwFCFm9JWSiL1puIVfjpEgNF2dSoy?usp=sharing 
    	cd ..	
		;;
	modelnet40_c)
		mkdir -p data/modelnet40_c
		cd data/modelnet40_c
		python ../../gdrivedl.py https://drive.google.com/drive/folders/10YeQRh92r_WdL-Dnog2zQfFr03UW4qXX?usp=sharing 
    	cd ../..
		;;
	modelnet40)
		wget --no-check-certificate https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
		unzip modelnet40_ply_hdf5_2048.zip
		mv modelnet40_ply_hdf5_2048 data
		rm -r modelnet40_ply_hdf5_2048.zip
    		wgetgdrive 1jXe7UR6He-pV3B7vIxMAjEt63Vhy1bV8 tmp/modelnet40_ply_hdf5_2048_valid_small.zip
		unzip -o tmp/modelnet40_ply_hdf5_2048_valid_small.zip
		mv modelnet40_ply_hdf5_2048_valid_small/* data/modelnet40_ply_hdf5_2048/
		rm -r modelnet40_ply_hdf5_2048_valid_small
		wget http://modelnet.cs.princeton.edu/ModelNet40.zip
		unzip ModelNet40.zip
		mv ModelNet40 data
		rm -r ModelNet40.zip
		rm -rf modelnet40_ply_hdf5_2048
		;;
	mesh)
		wget --no-check-certificate http://modelnet.cs.princeton.edu/ModelNet40.zip
		unzip ModelNet40.zip
		mv ModelNet40 data
		rm -r ModelNet40.zip
		;;
    	*)
    		echo "unknow argument $1" # unknown argument
    		;;
esac
rm -r tmp
