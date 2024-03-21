import argparse
import os
import shutil
from analysis_util import get_best_trial, copy_directory


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str,
                        # default='recommend_V2_NN/outputs/nn_outputs',
                        help=('the directory that stores all trained nn_outputs'))

    parser.add_argument('--output_dir', type=str,
                        # default='path/to/save/outputs',
                        help=('directory to save the best model.'))

    args = parser.parse_args()

    best_trial = get_best_trial(args.model_dir)

    best_model_dir = os.path.join(args.output_dir, 'best_model')

    if os.path.exists(best_model_dir):
        for file_name in os.listdir(best_model_dir):
            file_path = os.path.join(best_model_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print('Have cleaned up the directory!')
    else:
        os.makedirs(best_model_dir)


    copy_directory(best_trial, best_model_dir)

    # shutil.copytree(best_trial, best_model_dir)

    # print(os.listdir(os.path.join(best_model_dir, 'model')))

if __name__ == '__main__':
    main()