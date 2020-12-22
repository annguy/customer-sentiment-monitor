from datetime import datetime
from datetime import timedelta
import pickle
import os
import numpy as np
from tqdm import tqdm
from time import sleep
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, precision_score, recall_score


def add_week_to_cwdate(date, weeks=0):
    """Add a number of weeks to a date in CW in agreement with ISO8601.

    Parameters
    ----------
    date: str
        The date. Format: YYYY-WW (according to ISO8601)
    weeks: int
        Number of weeks to add. Default: 0

    Return
    ------
    str
        date + weeks. Format: YYYY-WW (according to ISO8601)
    """

    # Convert from cw to date
    original_date = cw_to_date(date)
    # Add weeks
    target_date = original_date + timedelta(days=7 * weeks)
    # Convert from date to cw
    target_date_cw = date_to_cw(target_date)
    return target_date_cw


def date_to_cw(date):
    """Convert datetime.date object to calendar week string (according to ISO8601).

    Parameter
    ---------
    date: datetime.date
        The date.

    Return
    ------
    str
        The calendar week. Format: YYYY-WW (according to ISO8601)
    """

    iso_cal = date.isocalendar()
    if iso_cal[1] < 10:
        return str(iso_cal[0]) + '-0' + str(iso_cal[1])
    else:
        return str(iso_cal[0]) + '-' + str(iso_cal[1])


def cw_to_date(cw):
    """Convert calendar week string (according to ISO8601) to datetime.date object.

    Parameter
    ---------
    cw : str
        The calendar week.  Format: YYYY-WW (according to ISO8601)

    Return
    ------
    datetime.date
        Date object of the respective week (Monday, 00:00)
    """

    return datetime.strptime(cw + '-1', '%G-%V-%u')


def calculate_recall_at_k(y_prob, y, k_max):
    """Calculate recall at k, where k varies from 1 to k_max. For each k:
    treat the top (with higher probability) k recommendations as the positive
    predictions, then calculate tp and recall.
    This function works correctly only for one pred_time.

    Parameters
    ----------
    y_prob: numpy.ndarray
        Predicted probabilities (num_of_samples)
    y: numpy.ndarray
        True/targer values (num_of_samples)
    k_max: int
        The max value of k

    Returns
    -------
    list
        A list of k. len()=k_max
    list
        A list of recalls. len()=k_max
    float
        The average recalls
    list
        A list of tps. len()=k_max
    list
        A list of flags. len()=k_max
    """

    # Init
    list_k, list_recall_at_k, list_num_tps, list_num_flags = list(range(1, k_max + 1)), [], [], []
    # Calculate recalls
    for k in range(k_max):
        # Calculate number of tp and number of flag
        num_tps, num_flags = np.sum(y[y_prob.argsort()[::-1][0:k]]), np.sum(y)
        list_num_tps.append(num_tps), list_num_flags.append(num_flags)
        # Append recall if num_flags not zero, else 0
        if int(num_flags) != 0:
            list_recall_at_k.append(num_tps / num_flags)
        else:
            list_recall_at_k.append(0)
    # Calculate average_recalls_at_k
    avg_recall = np.mean(list_recall_at_k)
    return list_k, list_recall_at_k, avg_recall, list_num_tps, list_num_flags


def create_model_info(clf, config, info, y_test, y_pred, y_prob):
    """Create dict with the experiment/model info.

    Parameters
    ----------
    clf:
        Trained classifier object
    config: dict
        A dict of configurations
    info: pd.DataFrame
        A DataFrame of trials
    y_test: numpy.ndarray
        A array containing the target labels from test set
    y_pred: numpy.ndarray
        A array containing the predicted labels from test set
    y_prob: numpy.ndarray
        A array containing the predicted probabilities from test set

    Return
    ------
    dict
        A dictionary of the experiment/model info
    """

    # Define binary parameters
    labels = [0, 1]
    average = 'binary'
    # Calculate confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels=labels)
    # Create model_info dict
    model_info = {
        'model': pickle.dumps(clf),
        'info': pickle.dumps(info),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': np.round(y_prob, 6),
        'balanced_accuracy': round(balanced_accuracy_score(y_test, y_pred), 4),
        'confusion_matrix': conf_mat,
        'f1_score_weighted': round(f1_score(y_test, y_pred, average=average, labels=labels), 4),
        'precision_weighted': round(precision_score(y_test, y_pred, average=average, labels=labels), 4),
        'recall_weighted': round(recall_score(y_test, y_pred, average=average, labels=labels), 4),
        'f1_score': np.round(f1_score(y_test, y_pred, average=average, labels=labels), 4),
        'precision': np.round(precision_score(y_test, y_pred, average=average, labels=labels), 4),
        'recall': np.round(recall_score(y_test, y_pred, average=average, labels=labels), 4),
        'date_created': datetime.now(),
        'config': config
    }
    return model_info


def cal_week_gen(start_cal_week, stop_cal_week):
    """Generator to conveniently iterate through calendar weeks.

    Like most iterators, includes the start value, but excludes the stop value.
    Calendar weeks according to ISO8601.

    Parameters
    ----------
    start_cal_week: str
        First value of the iteration (included). Format: YYYY-WW (according to ISO8601)
    stop_cal_week: str
        Last value of the iteration (excluded). Format: YYYY-WW (according to ISO8601)

    Yield
    -----
    str
        Current calendar week. Format: YYYY-WW (according to ISO8601)
    """

    # Convert weeks to datetime
    week_delta = timedelta(weeks=1)
    current_cal_week_date = cw_to_date(start_cal_week)
    stop_cal_week_date = cw_to_date(stop_cal_week)
    # Iterate through calendar weeks
    while date_to_cw(current_cal_week_date) < date_to_cw(stop_cal_week_date):
        yield date_to_cw(current_cal_week_date)
        current_cal_week_date += week_delta


def filter_and_split_feature_matrix_by_cal_week(feature_matrix, cal_week, config):
    """Filter and split the feature matrix for the weekly analysis.

    Removes all samples from the feature matrix with pred_time > cal_week.
    Split the remaining samples to a train set with pred_time < cal_week - lookahead_window_len + 1
    and a test set with pred_time == cal_week.

    Parameters
    ----------
    feature_matrix: pd.DataFrame
        Feature matrix
    cal_week: str
        Week used for the prediction. Format: YYYY-WW
    config: dict
        A dict of configurations

    Returns
    -------
    train_feature_matrix: pd.DataFrame
        Feature matrix containing the train set
    test_feature_matrix: pd.DataFrame
        Feature matrix containing the test set
    """

    train_feature_matrix = feature_matrix[feature_matrix.pred_time < \
                                          add_week_to_cwdate(cal_week, weeks=-config['lookahead_window_len'] + 1)]
    test_feature_matrix = feature_matrix[feature_matrix.pred_time == cal_week]
    return train_feature_matrix, test_feature_matrix


def update_log(path, obj, pred_time):
    """Update the saved object. Save the current pred_time object
    and delete the previous pred_time object.

    Parameters
    ----------
    path: str
        A path used to save pickle file
    obj:
        The object that should be saved
    pred_time: str
        The prediction time
    """

    # Save the current object
    with open(path.format(pred_time), 'wb') as f:
        pickle.dump(obj, f)
    # Delete the previous object
    if os.path.exists(path.format(add_week_to_cwdate(pred_time, weeks=-1))):
        os.remove(path.format(add_week_to_cwdate(pred_time, weeks=-1)))


def convert_feature_matrix_to_lstm_format(feature_matrix):
    """ Convert feature matrix to lstm input format.
    (num_features) -> (num_pred_time, num_features per pred_time)

    Parameters
    ----------
    feature_matrix: pd.DataFrame(num_samples, 323)
        The DataFrame with feature matrix containing:
        'customer', 'pred_time', 'escalation_flag' and 320 feature columns.

    Return
    -------
    pd.DataFrame (num_samples, 4)
        The DataFrame with feature matrix containing 4 columns:
        'customer', 'pred_time', 'feat' and 'label'.
    """

    # Extract feature columns
    # feat_cols (len(feat_names)=10, len(feat_names[0])=32)
    feat_cols = [sorted([col for col in feature_matrix.columns if '0{}'.format(i) in col]) for i in range(10)]
    # Init feat column
    feature_matrix.loc[:, 'feat'] = None
    # Use sleep to correct visualization of tqdm bar
    sleep(0.1)
    # Iterate though rows
    for ind in tqdm(range(feature_matrix.shape[0])):
        row = feature_matrix.iloc[ind]
        feat = np.zeros((10, 32))
        # Iterate through weeks
        for i in range(10):
            feat[i, :] = row[feat_cols[i]]
        feature_matrix.at[ind, 'feat'] = feat
    sleep(0.1)
    # Preprocessed feature_matrix
    feature_matrix_prep = feature_matrix[['customer', 'pred_time', 'feat', 'escalation_flag']]
    feature_matrix_prep = feature_matrix_prep.rename(columns={'escalation_flag': 'label'})
    return feature_matrix_prep
