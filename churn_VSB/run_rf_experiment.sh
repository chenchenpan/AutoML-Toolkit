export H=/home/chenchenpan/Projects
export DIR=$H/AutoML-Toolkit
export DATA_DIR=$DIR/churn_VSB/data/raw_data
export ENCODED_DATA_DIR=$DIR/churn_VSB/data/encoded_data

#python $DIR/encode_data.py \
#       --data_dir $DATA_DIR \
#       --output_dir $ENCODED_DATA_DIR \
#       --metadata_dir $DATA_DIR\
#       --train_file $DATA_DIR/vsb_train.tsv \
#       --dev_file $DATA_DIR/vsb_dev.tsv\
#       --test_file $DATA_DIR/vsb_test.tsv \
#       --use_text_features False \


export OUTPUT_DIR=$DIR/churn_VSB/outputs/rf_outputs
export SEARCH_SPACE=$DIR/churn_VSB/search_space

python $DIR/experiments.py \
       --encoded_data_dir $ENCODED_DATA_DIR \
       --search_space_dir $SEARCH_SPACE \
       --search_space_filename rf_search_space.json \
       --output_dir $OUTPUT_DIR \
       --task_type classification \
       --num_classes 2 \
       --num_outputs 1 \
       --model_type random_forest \
       --num_trials 2


