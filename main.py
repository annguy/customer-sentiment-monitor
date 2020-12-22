import pickle
import os
from src.config_parser import parse_config
import src.train_LSTM
import src.train_Ensemble
from src.general_helper_functions import convert_feature_matrix_to_lstm_format


def main(ini_file_path, data_path):
    """

    Parameter
    ----------
    ini_file_path: str
        The path to config.ini
    data_path: str
        The path to feature matrix
    """

    # Parse configurations
    config = parse_config(ini_file_path)
    # load data
    if config['model_name'] == 'LSTM':
        if os.path.exists(data_path + 'raw/feature_matrix_LSTM.pickle'):
            print('load file data for LSTM processing')
            with open(data_path + 'raw/feature_matrix_LSTM.pickle', 'rb') as f:
                feature_matrix_LSTM = pickle.load(f)
        else:
            print('LSTM file not existing yet')
            with open(data_path + 'raw/feature_matrix.pickle', 'rb') as f:
                feature_matrix = pickle.load(f)
            print('transform feature matrix for LSTM processing')
            print('this might take a few minutes')
            feature_matrix_LSTM = convert_feature_matrix_to_lstm_format(feature_matrix)
            print('save dataframe for LSTM processing')
            with open(data_path + 'raw/feature_matrix_LSTM.pickle', 'wb') as f:
                pickle.dump(data_path + 'raw/feature_matrix_LSTM.pickle', f)
        # Weekly analysis
        classifier_dict = src.train_LSTM.analyze_weekly(feature_matrix_LSTM, config, data_path)
    else:
        with open(data_path + 'raw/feature_matrix.pickle', 'rb') as f:
            feature_matrix = pickle.load(f)
        # Weekly analysis
        classifier_dict = src.train_Ensemble.analyze_weekly(feature_matrix, config, data_path)
    # Save classifier_dict
    with open(data_path+"/results/results.pickle", 'wb') as f:
        pickle.dump(classifier_dict, f)


if __name__ == '__main__':
    INI_FILE_PATH = 'config.ini'
    DATA_PATH = 'data/'
    main(INI_FILE_PATH, DATA_PATH)
