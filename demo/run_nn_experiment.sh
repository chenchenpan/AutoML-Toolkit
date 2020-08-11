export DIR=$HOME/projects/Amplify-AutoML-Toolkit
export DATA_DIR=$DIR/demo/data/raw_data
export GLOVE_DIR=$DIR/resource/glove
export ENCODED_DATA_DIR=$DIR/demo/data/encoded_data_comb_glove

python $DIR/encode_data.py \
       --output_dir $ENCODED_DATA_DIR \
       --metadata_file $DATA_DIR/metadata_comb.json \
       --train_file $DATA_DIR/comb_train.tsv \
       --dev_file $DATA_DIR/comb_dev.tsv \
       --test_file $DATA_DIR/comb_test.tsv \
       --use_text_features True \
       --encode_text_with glove\
       --glove_file $GLOVE_DIR/glove.6B.50d.txt \
       --max_words 10000 \
       --max_sequence_length 50


export OUTPUT_DIR=$DIR/demo/outputs/nn_outputs
export SEARCH_SPACE=$DIR/demo/search_space/nn_search_space.json

python $DIR/experiments.py \
       --encoded_data_dir $ENCODED_DATA_DIR \
       --search_space_filepath $SEARCH_SPACE \
       --output_dir $OUTPUT_DIR \
       --task_type classification \
       --num_classes 2 \
       --model_type mlp \
       --num_trials 5


