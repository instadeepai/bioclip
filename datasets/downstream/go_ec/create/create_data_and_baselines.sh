sudo docker build -t deepfri .
git clone https://github.com/flatironinstitute/DeepFRI.git

echo "Running all DeepFRI features..."
sudo docker run --rm --gpus all -v $PWD/downstream_data_preprocessing/deepfri:/app deepfri "\
    python DeepFRI/preprocessing/PDB2distMap.py \
    -annot /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv \
    -seqres /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_sequences.fasta \
    -num_threads 32"

echo "Running all canonical features..."
sudo docker run --rm --gpus all -v $PWD/downstream_data_preprocessing/deepfri:/app deepfri "\
    python create_dataset.py --num-process 62" #create_protein_h5.py"

echo "Running baselines... cellular_component"
sudo docker run --rm --gpus all -v $PWD/downstream_data_preprocessing/deepfri:/app deepfri "\
    python run_inference_on_test_set.py \
    -lm /app/DeepFRI/trained_models/lstm_lm.hdf5 \
    --params-path /app/DeepFRI/trained_models/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_cellular_component_model_params.json \
    --checkpoint-path /app/DeepFRI/trained_models/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_cellular_component \
    --annot-fn /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv \
    --test-list /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_test.csv"

echo "Running baselines... biological_process"
sudo docker run --rm --gpus all -v $PWD/downstream_data_preprocessing/deepfri:/app deepfri "\
    python run_inference_on_test_set.py \
    -lm /app/DeepFRI/trained_models/lstm_lm.hdf5 \
    --params-path /app/DeepFRI/trained_models/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_2048_ca_10A_biological_process_model_params.json \
    --checkpoint-path /app/DeepFRI/trained_models/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_2048_ca_10A_biological_process \
    --annot-fn /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv \
    --test-list /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_test.csv"

echo "Running baselines... molecular_function"
sudo docker run --rm --gpus all -v $PWD/downstream_data_preprocessing/deepfri:/app deepfri "\
    python run_inference_on_test_set.py \
    -lm /app/DeepFRI/trained_models/lstm_lm.hdf5 \
    --params-path /app/DeepFRI/trained_models/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_molecular_function_model_params.json \
    --checkpoint-path /app/DeepFRI/trained_models/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_molecular_function \
    --annot-fn /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv \
    --test-list /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_test.csv"

echo "Running baselines... enzyme_commission"
sudo docker run --rm --gpus all -v $PWD/downstream_data_preprocessing/deepfri:/app deepfri "\
    python run_inference_on_test_set.py \
    -lm /app/DeepFRI/trained_models/lstm_lm.hdf5 \
    --params-path /app/DeepFRI/trained_models/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_enzyme_commission_model_params.json \
    --checkpoint-path /app/DeepFRI/trained_models/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_enzyme_commission\
    --annot-fn /app/DeepFRI/preprocessing/data/nrPDB-EC_2020.04_annot.tsv \
    --test-list /app/DeepFRI/preprocessing/data/nrPDB-EC_2020.04_test.csv"

echo "Uploading baseline results..."
sudo docker run --rm --gpus all -v $PWD/downstream_data_preprocessing/deepfri:/app deepfri "\
    python collect_results_and_upload.py"

# echo "Update dataset..."
# sudo docker run --rm --gpus all -v $PWD/downstream_data_preprocessing/deepfri:/app deepfri "\
#     python -m create_protein_h5.update_dataset_based_on_any_files_that_couldnt_be_run"


## Training command...
# python train_DeepFRI.py \
#     -gc GAT -pd 100 -ont mf -lm /app/DeepFRI/trained_models/lstm_lm.hdf5 --model_name GAT-PDB_MF \
#     --train_tfrecord_fn /app/data/PDB-GO/PDB_GO_train_ --valid_tfrecord_fn /app/data/PDB-GO/PDB_GO_valid_ \
#     --annot_fn /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv \
#     --test_list /app/DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_test.csv