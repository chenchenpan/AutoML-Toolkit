export ENCODED_DATA_DIR=/datadrive/projects/Amplify-AutoML-Toolkit/encoded_data
export SEARCH_SPACE=/datadrive/projects/Amplify-AutoML-Toolkit/Demo_data/search_space.json
export OUTPUT_DIR=/datadrive/projects/Amplify-AutoML-Toolkit/outputs



python experiments.py --encoded_data_dir $ENCODED_DATA_DIR \
					  --search_space_filepath $SEARCH_SPACE \
					  --output_dir $OUTPUT_DIR \
					  --task_type classification \
					  --num_classes 2 \
					  --model_type mlp \
					  --num_trials 3
