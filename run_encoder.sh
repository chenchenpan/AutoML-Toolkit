export DATA_DIR=$HOME/projects/Amplify-AutoML-Toolkit/Demo_data
export GLOVE_DIR=$HOME/projects/Predict-The-Success-of-Crowdfunding/glove
export OUTPUT_DIR=$HOME/projects/Amplify-AutoML-Toolkit/encoded_data_struc_only


python encoder.py --data_dir $DATA_DIR \
				  --output_dir $OUTPUT_DIR\
				  --metadata_file metadata_struc_only.json \
				  --train_file comb_train.tsv \
				  --dev_file comb_dev.tsv \
				  --test_file comb_test.tsv \
				  --use_text_features True \
				  --encode_text_with tfidf \
				  --max_words 20 \
				  --glove_file $GLOVE_DIR/glove.6B.50d.txt \
				  --max_sequence_length 5\
				  --embedding_dim 50




