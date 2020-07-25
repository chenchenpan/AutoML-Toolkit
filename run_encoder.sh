export DATA_DIR=/datadrive/projects/Amplify-AutoML-Toolkit/Demo_data
export GLOVE_DIR=/datadrive/projects/glove
export OUTPUT_DIR=/datadrive/projects/Amplify-AutoML-Toolkit/encoded_data


python encoder.py --data_dir $DATA_DIR \
				  --output_dir $OUTPUT_DIR\
				  --metadata_file metadata_comb.json \
				  --train_file comb_train.tsv \
				  --dev_file comb_dev.tsv \
				  --test_file comb_test.tsv \
				  --use_text_features True \
				  --encode_text_with glove\
				  --glove_file $GLOVE_DIR/glove.6B.100d.txt \
				  --max_words 20\
				  --max_sequence_length 5\
				  --embedding_dim 100




