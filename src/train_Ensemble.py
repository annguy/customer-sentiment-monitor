import pickle
import warnings
import time
import os
from copy import deepcopy
import optuna
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from src.general_helper_functions import add_week_to_cwdate, calculate_recall_at_k, \
    create_model_info, cal_week_gen, filter_and_split_feature_matrix_by_cal_week, \
    update_log

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def sampling(x, y, ratio, flag='over'):
    """Sample data set. There are 3 available options:
    random oversample, random undersample or SMOTE.

    Parameters
    ----------
    x: pd.DataFrame
        A DataFrame of the features
    y: pd.DataFrame
        A DataFrame of the labels
    ratio: float
        The sampling ratio. The ratio is equal to the number of
        samples in the minority class over the number of samples
        in the majority class after resampling.
    flag: str {'over', 'over_SMOTE', 'under'}, optional
        The sampling strategy. Default: 'over'

    Returns
    -------
    pd.DataFrame
        A DataFrame of the sampled features
    pd.DataFrame
        A DataFrame of the sampled labels
    """

    # Init sampler
    if flag == 'over':
        sampler = RandomOverSampler(sampling_strategy=ratio, random_state=0)
    elif flag == 'over_SMOTE':
        sampler = SMOTE(sampling_strategy=ratio, random_state=0)
    elif flag == 'under':
        sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
    else:
        return None
    # Fit and apply the transform
    x, y = sampler.fit_resample(x, y)
    return x, y


def create_list_dataset(feature_matrix):
    """Convert feature_matrix DataFrame into list of feature_matrix DataFrames,
    where each list element (DataFrame) belongs to the unique pred_time.

    Parameter
    ---------
    feature_matrix: pd.DataFrame (num_samples, 323)
        The DataFrame with feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 feature columns.

    Returns
    -------
    list
        A list of DataFrames with features
    list
        A list of DataFrames with labels
    """

    list_x, list_y = [], []
    # Iterate through pred_time
    for pred_time in feature_matrix.pred_time.unique():
        # Slice DataFrame
        feature_matrix_slice = feature_matrix[feature_matrix.pred_time == pred_time]
        # Split feat_mat into features x and labels y
        list_x.append(feature_matrix_slice.drop(columns=["escalation_flag", "pred_time", "customer"]))
        list_y.append(feature_matrix_slice["escalation_flag"])
    return list_x, list_y


def predict(x, clf):
    """Use trained classifier to predict labels and probabilities.

    Parameters
    ----------
    x: pd.DataFrame
        A DataFrame containing features
    clf:
        Trained classifier object with implemented *predict* and
        *predict_prob* methods

    Returns
    -------
    np.array
        A array containing the predicted labels
    np.array
        A array containing the predicted probabilities
    """

    y_pred, y_prob = clf.predict(x), clf.predict_proba(x)
    return y_pred, y_prob


def get_columns(feature_matrix, feature_type):
    """Get feature columns from feature_matrix.

    Parameters
    ----------
    feature_matrix: pd.DataFrame (num_samples, 323)
        The DataFrame with feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 feature columns.
    feature_type: str {'ent' or 'log'}
        The type of features
    Return
    ------
    List
        A list of columns
    """

    if feature_type == 'enterprise':
        cols_type_1 = [col for col in feature_matrix.columns if col[0:3] == 'ent']
        return cols_type_1

    elif feature_type == 'log':
        cols_type_2 = [col for col in feature_matrix.columns if col[0:3] == 'log']
        return cols_type_2


def filter_by_feature_type(feature_matrix, feature_type):
    """Filter feature matrix by feature_type.
    If type is equal to 'both', then just return feature_matrix.
    Otherwise, return filtered feature matrix.

    Parameters
    ----------
    feature_matrix: pd.DataFrame (num_samples, 323)
        The DataFrame with feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 feature columns.
    feature_type: str {'ent', 'log', 'both_feature_types'}
        Type of features that should be left

    Return
    ------
    DataFrame
        Feature matrix with chosen features
    """

    if feature_type != 'both_feature_types':
        # Define cols which are not features
        not_feat_cols = ['pred_time', 'escalation_flag', 'customer']
        # Define the cols which are features
        cols = get_columns(feature_matrix, feature_type)
        # Filter feature_matrix by cols
        feature_matrix = feature_matrix.loc[:, cols + not_feat_cols]
    return feature_matrix


def stacking_model_init(clf, feature_matrix):
    """Init stacking classifier. Using StackingClassifier
    can be treated as late fusion.

    Parameters
    ----------
    clf: sklearn.ensemble
        A classifier
    feature_matrix: pd.DataFrame (num_samples, 323)
        The DataFrame with feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 feature columns.

    Return
    ------
    sklearn.ensemble
        The initialized classifier
    """

    # Find feature cols
    cols_type_1 = get_columns(feature_matrix, 'enterprise')
    cols_type_2 = get_columns(feature_matrix, 'log')
    # Convert name into column number
    ind_type_1 = [ind for ind, col in enumerate(
        list(feature_matrix.drop(columns=['escalation_flag', 'pred_time', 'customer']).columns)) if
                  col in cols_type_1]
    ind_type_2 = [ind for ind, col in enumerate(
        list(feature_matrix.drop(columns=['escalation_flag', 'pred_time', 'customer']).columns)) if
                  col in cols_type_2]
    # Init stacking classifier
    clf = stacked_predictor(clf, ind_type_1, ind_type_2)
    return clf


def model_init(feature_matrix, params, config):
    """Init model.

    Parameters
    ----------
    feature_matrix: pd.DataFrame (num_samples, 323)
        The DataFrame with feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 feature columns.
    params: dict
        The parameters for tuning
    config: dict
        The configurations

    Return
    -------
    sklearn.ensemble
        The classifier
    """

    # Init classifier
    if config['model_name'] == 'XGBoost':
        clf = xgb.XGBClassifier()
    elif config['model_name'] == 'RandomForestClassifier':
        clf = RandomForestClassifier()
    # Set parameters
    clf.set_params(**params)
    # Init StackingClassifier
    if config['late_fusion_flag']:
        clf = stacking_model_init(clf, feature_matrix)
    return clf


def remove_keys_from_dict(dictionary, keys):
    """Remove keys from dict.

    Parameters
    ----------
    dictionary: dict
        The input dictionary
    keys: list
        The list of keys that should be removed
    """

    # Copy dictionary
    dictionary_updated = dictionary.copy()
    try:
        [dictionary_updated.pop(key) for key in keys]
    except:
        print("Error: No ratio and sampling strategy parameters")
    return dictionary_updated


def train_on_one_set(feature_matrix, x_train, y_train, x_val, y_val, config, params):
    """Train on one set of parameter.

    Parameters
    ----------
    feature_matrix: pd.DataFrame (num_samples, 323)
        The DataFrame with feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 features columns.
    x_train: list
        A list of the train features
        (len(x_train)=num_train_week, x_train.shape=(num_train_samples, num_features))
    y_train: list
        A list of the train labels
        (len(y_train)=num_train_week, y_train.shape=(num_train_samples))
    x_val: list
        A list of the validation features
        (len(x_val)=num_val_week, x_val.shape=(num_val_samples, num_features))
    y_val: list
        A list of the validation labels
        (len(y_val)=num_val_week, y_val.shape=(num_val_samples))
    config: dict
        The configurations
    params: dict
        A dictionary of the parameters for tuning

    Returns
    -------
    sklearn.ensemble
        The classifier
    float
        The evaluation metric
    dict
        The parameters
    """

    # Remove keys from dict
    params_model = remove_keys_from_dict(params, keys=['ratio', 'sampling_strategy'])
    # Init classifier
    clf = model_init(feature_matrix, params_model, config)
    # x_train_flat (num_samples, num_features)
    # y_train_flat (num_samples)
    x_train_flat, y_train_flat = pd.concat(x_train), pd.concat(y_train)
    # Sample
    if all(key in params for key in ("ratio", "sampling_strategy")):
        # x_train_flat (num_samples after sampling, num_features)
        # y_train_flat (num_samples after sampling)
        x_train_flat, y_train_flat = sampling(x_train_flat, y_train_flat, params['ratio'],
                                              flag=params['sampling_strategy'])
    #  Fit classifier on sampled data set
    clf.fit(x_train_flat, y_train_flat)
    # Calculate avg_val_recall
    _, _, _, _, avg_val_recall = calculate_recall_at_k_time_series(x_val, y_val, clf)
    return clf, avg_val_recall, params


def output_report(x, y, clf, test_flag=False):
    """Output/print the confusion matrix, average recalls at 100 and
    positions of escalation_flag.

    Parameters
    ----------
    x: list
        A list of the features
        (len(x)=num_week, x.shape=(num_samples, num_features))
    y: list
        A list of the labels
        (len(y)=num_week, y_val.shape=(num_samples))
    clf: sklearn.ensemble
        The classifier
    test_flag: bool, optional
        Output positions of escalation flags if the test set is used. Default: False
    """

    # Calculate recall at k time series
    y, y_prob, y_pred, recall_at_k, avg_recall = calculate_recall_at_k_time_series(x, y, clf)
    # Output confusion_matrix
    print("Confusion matrix: \n", confusion_matrix(y, y_pred))
    # Output recall_at_k
    print("Recall at {}, {}, {}, {}, {}: ".format(5, 10, 20, 50, 100), end=' ')
    for top in [5, 10, 20, 50, 100]:
        if top == 100:
            print(str(round(recall_at_k[top - 1], 2)) + " accordingly")
        else:
            print("{}, ".format(round(recall_at_k[top - 1], 2)), end='')
    print("Average recalls over 100: ", round(avg_recall, 2))
    if test_flag:
        _, recall_at_k, _, _, _ = calculate_recall_at_k(y_prob[:, 1], y, k_max=y.shape[0])
        print(
            "Positions of escalation flags: ", ([1] if recall_at_k[0] != 0 else []) +
                                               [i + 1 for i in range(1, len(recall_at_k)) if
                                                recall_at_k[i] != recall_at_k[i - 1]])


def suggest_parameters(trial, config):
    """Init/suggest parameters for optuna.

    Parameters
    ----------
    trial:
        The object of the optuna
    config: dict
        The configurations

    Return
    ------
    dict
        The parameters for optuna
    """

    # Get parameters from config
    parameters = config['params_' + config['model_name']]
    # Init parameters for optuna
    optuna_parameters = dict()
    for key in parameters.keys():
        if parameters[key][0] == 'int':
            optuna_parameters[key] = trial.suggest_int(key, parameters[key][1], parameters[key][2])
        elif parameters[key][0] == 'uniform':
            optuna_parameters[key] = trial.suggest_uniform(key, parameters[key][1], parameters[key][2])
        elif parameters[key][0] == 'categorical':
            optuna_parameters[key] = trial.suggest_categorical(key, parameters[key][1])
        elif parameters[key][0] == 'loguniform':
            optuna_parameters[key] = trial.suggest_loguniform(key, parameters[key][1], parameters[key][2])
    return optuna_parameters


def objective(trial, feature_matrix, x_train, y_train, x_val, y_val, config, data_path):
    """Define objective function used for optuna trials.

    Parameters
    ----------
    trial: optuna.trial
    feature_matrix: pd.DataFrame (num_samples, 323)
        The DataFrame with feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 features columns.
    x_train: list
        A list of the train features
        (len(x_train)=num_train_week, x_train.shape=(num_train_samples, num_features))
    y_train: list
        A list of the train labels
        (len(y_train)=num_train_week, y_train.shape=(num_train_samples))
    x_val: list
        A list of the validation features x_val
        (len(x_val)=num_val_week, x_val.shape=(num_val_samples, num_features))
    y_val: list
        A list of the validation labels
        (len(y_val)=num_val_week, y_val.shape=(num_val_samples))
    config: dict
        The configurations
    data_path: str
        The path used to save and load data

    Return
    ------
    float
        The metric used for optimization
    """

    # Init params for trial
    params = suggest_parameters(trial, config)
    # Calculate avg_val_recall
    clf, avg_val_recall, _ = train_on_one_set(feature_matrix, x_train, y_train, x_val, y_val, config, params)
    # Save a trained model to a file.
    with open(data_path + 'interim/trial_{}.pickle'.format(trial.number), 'wb') as f:
        pickle.dump(clf, f)
    return avg_val_recall


def calculate_recall_at_k_time_series(x, y, clf, k_max=100):
    """Calculate recall at k for time series data. For each pred_time:
    use calculate_recall_at_k. Then combine outputs from calculate_recall_at_k.

    Parameters
    ----------
    x: list
        A list of the features
        (len(x)=num_week, x.shape=(num_samples, num_features))
    y: list
        A list of the labels
        (len(y)=num_week, y_val.shape=(num_samples))
    clf: sklearn.ensemble
        A classifier
    k_max: int, optional
        The max number of top recommendations. Default: 100

    Returns
    -------
    list
        A list of recall at k
    float
        The average recall at k
    float
        Loss value
    np.ndarray
        The array of target values
    np.ndarray
        The array of predicted probabilities
    np.ndarray
        The array of predicted values
    """

    # Init
    y_target, y_pred, y_prob = None, None, None
    tps, flags = np.zeros(k_max), np.zeros(k_max)
    # Iterate through pred_time
    for pred_time in range(len(x)):
        # Predict
        y_pred_set, y_prob_set = predict(x[pred_time], clf)
        y_set = y[pred_time].to_numpy()
        # Evaluate
        _, _, _, num_of_tp, num_of_flag = calculate_recall_at_k(y_prob_set[:, 1], y_set, k_max=100)
        # tps (k_max)
        tps += np.array(num_of_tp)
        # flags (k_max)
        flags += np.array(num_of_flag)
        # Concatenate values
        y_target, y_prob, y_pred = (y_set, y_prob_set, y_pred_set) if y_target is None \
            else (np.concatenate((y_target, y_set)),
                  np.concatenate((y_prob, y_prob_set)),
                  np.concatenate((y_pred, y_pred_set)))
    return y_target, y_prob, y_pred, np.nan_to_num(tps / flags), np.mean(np.nan_to_num(tps / flags))


def tpe_sampler_search(feature_matrix, x_train, y_train, x_val, y_val, config, data_path):
    """Use optuna.TPESampler to tune hyperparameters.

    Parameters
    ----------
    feature_matrix: pd.DataFrame (num_samples, 323)
        The DataFrame with feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 features columns.
    x_train: list
        A list of the train features
        (len(x_train)=num_train_week, x_train.shape=(num_train_samples, num_features))
    y_train: list
        A list of the train labels
        (len(y_train)=num_train_week, y_train.shape=(num_train_samples))
    x_val: list
        A list of the validation features x_val
        (len(x_val)=num_val_week, x_val.shape=(num_val_samples, num_features))
    y_val: list
        A list of the validation labels
        (len(y_val)=num_val_week, y_val.shape=(num_val_samples))
    config: dict
        The configurations
    data_path: str
        The path used to save and load data

    Returns
    -------
    sklearn.ensemble
        The best classifier
    DataFrame
        A DataFrame of trials
    """

    # Init sampler and n_trials
    sampler = optuna.samplers.TPESampler()
    n_trials = config['number_of_trials']
    # Create study
    study = optuna.create_study(sampler=sampler, direction='maximize')
    # Disable output
    optuna.logging.disable_default_handler()
    # Optimize
    study.optimize(lambda trial: objective(trial, feature_matrix, x_train, y_train,
                                           x_val, y_val, config, data_path),
                   n_trials=n_trials)
    # Init model with best parameters
    print("Best trial: ", study.best_trial.number)
    print("Best parameters: ", study.best_params)
    # Load the best model.
    with open(data_path + 'interim/trial_{}.pickle'.format(study.best_trial.number), 'rb') as f:
        clf = pickle.load(f)
    # Delete all trials
    for trial_num in range(config['number_of_trials']):
        if os.path.exists(data_path + 'interim/trial_{}.pickle'.format(trial_num)):
            os.remove(data_path + 'interim/trial_{}.pickle'.format(trial_num))
    print("***Train***")
    output_report(x_train, y_train, clf)
    print("***Validation***")
    output_report(x_val, y_val, clf)
    # Remove keys from dict
    best_params_model = remove_keys_from_dict(study.best_params, keys=['ratio', 'sampling_strategy'])
    best_clf = model_init(feature_matrix, best_params_model, config)
    return best_clf, study.trials_dataframe()


def tune_hyperparameter(feature_matrix_train_val, config, data_path):
    """Tune hyperparameter and return the model with the best parameter.

    Parameters
    ----------
    feature_matrix_train_val: pd.DataFrame (num_samples, 323)
        The DataFrame with train and validation feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 features columns.
    config: dict
        The configurations
    data_path: str
        The path used to save and load data

    Returns
    -------
    sklearn.ensemble
        The best classifier
    DataFrame
        The info about hyperparameter tuning
    """

    # Split into train and validation set with (lookahead_window_len week-1) gap
    # Format: YYYY-WW (according to ISO8601)
    last_val_week = feature_matrix_train_val.pred_time.max()
    # Format: YYYY-WW (according to ISO8601)
    first_val_week = add_week_to_cwdate(last_val_week, weeks=-config['num_of_val_weeks'] + 1)
    # Format: YYYY-WW (according to ISO8601)
    last_train_week = add_week_to_cwdate(first_val_week, weeks=-config['lookahead_window_len'])
    # Split into train and validation DataFrames
    feature_matrix_train, feature_matrix_val = feature_matrix_train_val[
                                                   feature_matrix_train_val.pred_time <= last_train_week], \
                                               feature_matrix_train_val[
                                                   feature_matrix_train_val.pred_time >= first_val_week]
    print("Train set from {} to {}".format(feature_matrix_train.pred_time.min(), feature_matrix_train.pred_time.max()))
    print("Validation set from {} to {}".format(feature_matrix_val.pred_time.min(), feature_matrix_val.pred_time.max()))
    # Split features and labels and drop unnecessary columns
    # x_train (len(x_train)=num_train_week, x_train.shape=(num_train_samples, num_features))
    # y_train (len(y_train)=num_train_week, y_train.shape=(num_train_samples))
    x_train, y_train = create_list_dataset(feature_matrix_train)
    # x_val (len(x_val)=num_val_week, x_val.shape=(num_val_samples, num_features))
    # y_val (len(y_val)=num_val_week, y_val.shape=(num_val_samples))
    x_val, y_val = create_list_dataset(feature_matrix_val)
    # TPESampler search
    best_clf, info = tpe_sampler_search(feature_matrix_train, x_train, y_train,
                                        x_val, y_val, config, data_path)
    return best_clf, info


def stacked_predictor(clf, feature_type_1_ind, feature_type_2_ind):
    """Stacking predictor. Combine 2 classifiers by means of logistic regression.

    Parameters
    ----------
    clf: sklearn.ensemble
        Classifier
    feature_type_1_ind: list
        A list of indexes of the type_1 features
    feature_type_2_ind: list
        A list of indexes of the type_2 features

    Return
    ------
    sklearn.ensemble.StackingClassifier
        StackingClassifier
    """

    # Define (sub)classifiers
    pipe1 = make_pipeline(ColumnSelector(cols=feature_type_1_ind),
                          deepcopy(clf))
    pipe2 = make_pipeline(ColumnSelector(cols=feature_type_2_ind),
                          deepcopy(clf))
    # Init StackingClassifier
    stacked_clf = StackingClassifier(classifiers=[pipe1, pipe2],
                                     meta_classifier=LogisticRegression(class_weight='balanced'),
                                     use_probas=True,
                                     average_probas=False
                                     )
    return stacked_clf


def analyze_weekly(feature_matrix, config, data_path):
    """Train a classifier using all data previous of week N for training and then
       predict escalation flag for week N+1 and N+2.
       Assuming that week N is the last week for which all data is available.

    Parameters
    ----------
    feature_matrix: pd.DataFrame (num_samples, 323)
        The DataFrame with feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 feature columns.
    config: dict
        The configurations
    data_path: str
        The path used to save and load data

    Return
    -------
    classifier_dict : dict{week : (model_info)}
        Dictionary containing the trained models for each week
    """

    if config['model_name'] == 'XGBoost':
        print("XGBoost is used")
    elif config['model_name'] == 'RandomForestClassifier':
        print("RandomForestClassifier is used")
    if config['late_fusion_flag']:
        print("Late fusion is used")
    else:
        print("Early fusion is used")
    if config['feature_type'] == 'both_feature_types':
        print("Both type 1 and 2 features are used")
    elif config['feature_type'] == 'enterprise':
        print("Only enterprise features are used")
    elif config['feature_type'] == 'log':
        print("Only log features are used")
    # Format: YYYY-WW (according to ISO8601)
    first_pred_time = config['first_week']
    # Format: YYYY-WW (according to ISO8601)
    last_prediction_time = config['last_week']
    # Continue if cont_week is not None
    if config['cont_week'] == 'None':
        # Init classifier_dict
        classifier_dict = dict()
    else:
        # Format: YYYY-WW (according to ISO8601)
        load_time = add_week_to_cwdate(config['cont_week'], weeks=-1)
        # Load the last available pred_time
        with open(data_path + '/interim/results_{}.pickle'.format(load_time), 'rb') as f:
            classifier_dict = pickle.load(f)
        # Format: YYYY-WW (according to ISO8601)
        first_pred_time = config['cont_week']
    # Iterate through pred_time
    feature_matrix = filter_by_feature_type(feature_matrix, feature_type=config['feature_type'])
    # Iterate through weeks
    for pred_time in cal_week_gen(first_pred_time, last_prediction_time):
        print("*************************************************************")
        print("Prediction time: {}".format(pred_time))
        # Log start time to calculate the elapsed time later
        time_start = time.time()
        # Calculate the last used time for train_val data set
        # Format: YYYY-WW (according to ISO8601)
        last_time = add_week_to_cwdate(pred_time, weeks=-config['lookahead_window_len'] - \
                                                        config['fixed_window_in_weeks']) \
            if config['fixed_window_in_weeks'] != -1 \
            else add_week_to_cwdate(feature_matrix.pred_time.min(), weeks=-1)
        # Divide into train_val and test DataFrames
        feature_matrix_train_val, feature_matrix_test = filter_and_split_feature_matrix_by_cal_week(
            feature_matrix[feature_matrix.pred_time > last_time], pred_time, config)
        # Split features and labels and drop unnecessary columns
        # x_test (len(x_test)=1, x_test[0].shape=(num_samples, num_features))
        x_test, y_test = create_list_dataset(feature_matrix_test)
        # Tune hyperparameter
        classifier, info = tune_hyperparameter(feature_matrix_train_val, config, data_path)
        # x_train_val (len(x_train_val)=num_train_val_week, x_train_val.shape=(num_train_val_samples, num_features))
        # y_train_val (len(y_train_val)=num_train_val_week, y_train_val.shape=(num_train_val_samples))
        x_train_val, y_train_val = create_list_dataset(feature_matrix_train_val)
        # x_train_val_flat (num_train_val_samples, num_features)
        # y_train_val_flat (num_train_val_samples)
        x_train_val_flat, y_train_val_flat = pd.concat(x_train_val), pd.concat(y_train_val)
        # Sample
        try:
            # Extract best trial
            best_trial = info.value.argmax()
            # Extract ratio
            ratio = info.loc[best_trial, :].params_ratio
            # Extract sampling_strategy
            sampling_strategy = info.loc[best_trial, :].params_sampling_strategy
            # x_train_flat (num_train_val_samples after sampling, num_features)
            # y_train_flat (num_train_val_samples after sampling)
            x_train_val_flat, y_train_val_flat = sampling(x_train_val_flat, y_train_val_flat, ratio,
                                                          flag=sampling_strategy)
        except:
            print("Error: Failed to sample the data set")
        # Train classifier
        classifier.fit(x_train_val_flat, y_train_val_flat)
        # Make prediction on test set
        y_pred, y_prob = predict(x_test[0], classifier)
        # Create model info
        model_info = create_model_info(classifier,
                                       config, info,
                                       y_test[0], y_pred, y_prob)
        classifier_dict[pred_time] = model_info
        print("***Train and Validation***")
        output_report(x_train_val, y_train_val, classifier)
        print("***Test***")
        output_report(x_test, y_test, classifier, test_flag=True)
        # Update feature_matrix and (classifier_dict)
        update_log(data_path + '/interim/results_{}.pickle', classifier_dict, pred_time)
        time_diff = time.time() - time_start
        print("Time elapsed: {} minutes".format(round(time_diff / 60, 2)))
        print("*************************************************************")
    return classifier_dict


if __name__ == '__main__':
    pass
