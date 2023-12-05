echo "Downloading PeSTo source code..."
git clone https://github.com/LBM-EPFL/PeSTo.git

sudo docker build -t pesto .

echo "rsync takes around 5 days to download this... You can cancel, comment this, then run the command below as I have re-uploaded to the aws bucket for bio-clip"
bash PeSTo/data/rsyncPDB.sh

echo "Downloading raw PeSTo data from the bio-clip aws bucket..."
aws s3 cp s3://deepchain-research/bio_clip/pesto-data/all_biounits_1June2023.tar.gz $PWD/downstream_data_preprocessing/pesto --endpoint https://s3.kao.instadeep.io

echo "Processing all of the data and uploading chunks to the bio-clip bucket... (takes around 5 days on 100 CPUs)"
sudo docker run --rm --gpus all -v $PWD/downstream_data_preprocessing/deepfri:/app deepfri "\
    python create_data_chunked.py"

echo "Creating the h5 file from chunks, takes around 2 hours..."
sudo docker run --rm --gpus all -v $PWD/downstream_data_preprocessing/pesto:/app pesto "\
    python create_h5_from_chunks.py"

