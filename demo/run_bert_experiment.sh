export DIR=$HOME/projects/Amplify-AutoML-Toolkit
export RAW_DATA_DIR=$DIR/demo/data/raw_data
export ENCODED_DATA_DIR=$DIR/demo/data/bert_encoded_data

python $DIR/encode_bert_data.py \
       --output_dir $ENCODED_DATA_DIR \
       --train_file $RAW_DATA_DIR/comb_train.tsv \
       --dev_file $RAW_DATA_DIR/comb_dev.tsv \
       --test_file $RAW_DATA_DIR/comb_test.tsv \
       --text_col desc_clean \
       --label_col label


export OUTPUT_DIR=$DIR/demo/outputs/bert_outputs
export SEARCH_SPACE=$DIR/demo/search_space/bert_search_space.json
export BERT_DIR=$DIR/resource/bert/uncased_L-2_H-128_A-2

python $DIR/experiments.py \
    --encoded_data_dir $ENCODED_DATA_DIR \
    --search_space_filepath $SEARCH_SPACE \
    --output_dir $OUTPUT_DIR \
    --task_type classification \
    --num_classes 2 \
    --model_type bert \
    --bert_dir $BERT_DIR \
    --num_trials 5
