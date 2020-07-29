# export ENCODED_DATA_DIR=$HOME/projects/Amplify-AutoML-Toolkit/encoded_data
# export SEARCH_SPACE=$HOME/projects/Amplify-AutoML-Toolkit/Demo_data/nn_search_space.json
# export OUTPUT_DIR=$HOME/projects/Amplify-AutoML-Toolkit/nn_outputs



# python experiments.py --encoded_data_dir $ENCODED_DATA_DIR \
# 					  --search_space_filepath $SEARCH_SPACE \
# 					  --output_dir $OUTPUT_DIR \
# 					  --task_type classification \
# 					  --num_classes 2 \
# 					  --model_type mlp \
# 					  --num_trials 5


export ENCODED_DATA_DIR=$HOME/projects/Amplify-AutoML-Toolkit/encoded_data_struc_only
export SEARCH_SPACE=$HOME/projects/Amplify-AutoML-Toolkit/Demo_data/rf_search_space.json
export OUTPUT_DIR=$HOME/projects/Amplify-AutoML-Toolkit/rf_outputs



python experiments.py --encoded_data_dir $ENCODED_DATA_DIR \
					  --search_space_filepath $SEARCH_SPACE \
					  --output_dir $OUTPUT_DIR \
					  --task_type classification \
					  --num_classes 2 \
					  --model_type random_forest \
					  --num_trials 5


# export ENCODED_DATA_DIR=$HOME/projects/Amplify-AutoML-Toolkit/encoded_data
# export SEARCH_SPACE=$HOME/projects/Amplify-AutoML-Toolkit/Demo_data/lr_search_space.json
# export OUTPUT_DIR=$HOME/projects/Amplify-AutoML-Toolkit/lr_outputs



# python experiments.py --encoded_data_dir $ENCODED_DATA_DIR \
# 					  --search_space_filepath $SEARCH_SPACE \
# 					  --output_dir $OUTPUT_DIR \
# 					  --task_type classification \
# 					  --num_classes 2 \
# 					  --model_type logistic_regression \
# 					  --num_trials 5


# export ENCODED_DATA_DIR=$HOME/projects/Amplify-AutoML-Toolkit/encoded_data
# export SEARCH_SPACE=$HOME/projects/Amplify-AutoML-Toolkit/Demo_data/svm	_search_space.json
# export OUTPUT_DIR=$HOME/projects/Amplify-AutoML-Toolkit/svm_outputs



# python experiments.py --encoded_data_dir $ENCODED_DATA_DIR \
# 					  --search_space_filepath $SEARCH_SPACE \
# 					  --output_dir $OUTPUT_DIR \
# 					  --task_type classification \
# 					  --num_classes 2 \
# 					  --model_type svm \
# 					  --num_trials 5