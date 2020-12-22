from configparser import ConfigParser
from ast import literal_eval


def parse_config(config_path):
    """Parse config.ini.

    Parameter
    ----------
    config_path: str
        The path to config.ini

    Return
    -------
    dict
        A dictionary containing configuration parameters
    """

    parser = ConfigParser()
    parser.read(config_path)

    config_dict = {
        'lookahead_window_len': parser.getint('parameters', 'lookahead_window_len'),
        'num_of_val_weeks': parser.getint('parameters', 'num_of_val_weeks'),
        'fixed_window_in_weeks': parser.getint('parameters', 'fixed_window_in_weeks'),
        'feature_type': parser.get('parameters', 'feature_type'),
        'model_name': parser.get('parameters', 'model_name'),
        'first_week': parser.get('parameters', 'first_week'),
        'last_week': parser.get('parameters', 'last_week'),
        'late_fusion_flag': parser.getboolean('parameters', 'late_fusion_flag'),
        'number_of_trials': parser.getint('parameters', 'number_of_trials'),
        'patience': parser.getint('parameters', 'patience'),
        'cont_week': parser.get('parameters', 'cont_week'),
        'params_RandomForestClassifier': literal_eval(parser.get('parameters', 'params_RandomForestClassifier')),
        'params_XGBoost': literal_eval(parser.get('parameters', 'params_XGBoost')),
        'params_LSTM': literal_eval(parser.get('parameters', 'params_LSTM')),
        'n_startup_trials': parser.getint('parameters', 'n_startup_trials'),
        'n_warmup_steps': parser.getint('parameters', 'n_warmup_steps'),
        'interval_steps': parser.getint('parameters', 'interval_steps')
    }
    return config_dict


if __name__ == '__main__':
    pass
