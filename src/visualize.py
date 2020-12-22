import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pickle
from src.train_Ensemble import create_list_dataset


def convert_clf_dict_to_df(classifier_dict, feature_matrix, metrics, config):
    """Convert classifier_dict to DataFrame. Each row is timestamp/week.

    Parameters
    ----------
    classifier_dict : dict{week : (model_info)}
        Dictionary containing the trained models for each week
    feature_matrix: DataFrame
        Feature matrix. (num_samples, num_cols)
    metrics: list
        A list of metrics that should be extracted
    config: dict
        The configurations

    Return
    -------
    DataFrame
        The classifier DataFrame with extracted metrics (num_weeks, num_metrics)
    """

    # Create DataFrame with results
    df_data = pd.DataFrame()
    df_data['timestamp'] = pd.Series(list(classifier_dict))
    # Extract metrics
    for metric in metrics:
        df_data[metric] = pd.Series([classifier_dict[key][metric] for key in classifier_dict.keys()])
    df_data['# of samples'] = df_data.apply(lambda row: row.y_test.shape[0], axis=1)
    df_data['# of escalation flags'] = df_data.apply(lambda row: np.sum(row.y_test), axis=1)
    # Load X_test for ensembles
    if config['model_name'] != 'LSTM':
        df_data['X_test'] = None
        for ind, key in enumerate(classifier_dict.keys()):
            # x_test (len(x_test)=1, x_test[0].shape=(num_samples, num_features))
            x_test, _ = create_list_dataset(feature_matrix[feature_matrix.pred_time == key])
            # x_test[0] (num_samples, num_features)
            df_data.at[ind, 'X_test'] = x_test[0]
    # Rename columns
    # df_data (num_weeks, num_metrics)
    df_data = df_data.rename(columns={'timestamp': 'Timestamp'})
    # Reset indexes
    # df_data (num_weeks, num_metrics)
    df_data = df_data.reset_index(drop=True)
    return df_data


def extract_base_clf_probs(df_data):
    """Extract base classifiers probabilities and add to df_data.

    Parameter
    ----------
    df_data: DataFrame
        The classifier DataFrame (num_weeks, num_metrics)

    Return
    -------
    DataFrame
        The classifier DataFrame with base classifiers (num_weeks, num_metrics+2)
    """

    # Init cols
    df_data['y_prob_Base_1'], df_data['y_prob_Base_2'] = None, None
    # Extract probabilities
    for ind, key in enumerate(sorted(df_data.Timestamp.to_list())):
        # Load classifier
        clf = pickle.loads(df_data['model'][ind])
        # Predict using base classifiers
        df_data.at[ind, 'y_prob_Base_1'] = clf.clfs_[0].predict_proba(df_data.loc[ind, 'X_test'])[:, 1]
        df_data.at[ind, 'y_prob_Base_2'] = clf.clfs_[1].predict_proba(df_data.loc[ind, 'X_test'])[:, 1]
    return df_data


def calculate_recalls_at_k(df_data, k_max=150):
    """Calculate recalls at k for one week/timestamp.

    Parameters
    ----------
    df_data: DataFrame
        The classifier DataFrame
    k_max: int, optional
        The max value of k. Default: 150

    Returns
    -------
    DataFrame
        The total recalls
    int
        The number of total samples
    list
        A list of k. len()=k_max
    """

    # Calculate the number os samples and list_k
    num_of_samples = df_data['# of samples'].sum()
    list_k = list(range(1, k_max))
    # Define cols
    if all(key in df_data.columns for key in ("y_prob_Base_1", "y_prob_Base_2")):
        cols = ['y_prob_Final', "y_prob_Base_1", "y_prob_Base_2"]
        df_data = df_data.rename(columns={'y_prob': 'y_prob_Final'})
    else:
        cols = ['y_prob']
    # Iterate through cols
    total_recalls = {}
    for col in cols:
        recalls = []
        for k in list_k:
            try:
                tp = df_data.apply(lambda row: np.sum(row.y_test.to_numpy()[row[col].argsort()[::-1][:k]]), axis=1)
            except:
                tp = df_data.apply(lambda row: np.sum(row.y_test[row[col].argsort()[::-1][:k]]), axis=1)
            # Append recall
            recalls.append(tp.sum() / df_data['# of escalation flags'].sum() * 100)
        total_recalls[col[7:]] = recalls
    return total_recalls, num_of_samples, list_k


def plot_recall_at_k(recalls, list_k, num_of_samples=None, num_of_escalation_flags=None, title='', path='src/data/'):
    """Plot the recall at k

    Parameters
    ----------
    recalls: dict
        A dict of recalls
    list_k: list
        A list of k
    num_of_samples: int, optional
        A number of samples. Default: None
    num_of_escalation_flags: int, optional
        A number of escalation_flags. Default: None
    title: str, optional
        The plot title. Default: ''
    path: str, optional
        The path used for saving. Default: 'src/data/'
    """

    # Set size
    plt.rcParams["figure.figsize"] = (15, 6)
    # Set style
    sns.set()
    # Plot
    for key in recalls.keys():
        plt.plot(list_k, recalls[key])
    # Set legend
    plt.legend(recalls.keys())
    # Calculate max_recall
    max_recall = np.max([np.max(recalls[key]) for key in recalls.keys()])
    # Plot texts
    plt.text(-5, int(0.75 * max_recall), 'Total # of samples: {}'.format(num_of_samples), fontsize=20)
    plt.text(-5, int(0.65 * max_recall), 'Total # of escalation flags: {}'.format(num_of_escalation_flags), fontsize=20)
    # Set title
    plt.title(title, fontsize=28)
    # Set ticks
    plt.xticks([1] + list(np.arange(10, len(list_k)+10, 10)), fontsize=20)
    plt.yticks(fontsize=20)
    # Set labels
    plt.ylabel('Recall, %', fontsize=28)
    plt.xlabel('Top', fontsize=28)
    # Save as pdf
    plt.savefig(path+"Recall_at_k.pdf", bbox_inches="tight", format='pdf')
    plt.show()


def plot_pandas(ax, data, x, y, style=['bo--', 'cx--', 'gs--', 'rX--', 'yD--'],
                xlabel="", ylabel="", title=""):
    """Plot the DataFrame.

    Parameters
    ----------
    ax: matplotlib.axes
        The ax
    data: DataFrame
        A DataFrame of data
    x: str
        A column name used to x-axis
    y: str
        A column name used to y-axis
    style: list, optional
        A list of styles. Default: ['bo--', 'cx--', 'gs--', 'rX--', 'yD--']
    xlabel: str, optional
        The x-axis label. Default: ''
    ylabel: str, optional
        The y-axis label. Default: ''
    title: str, optional
        The title. Default: ''
    """

    # Set params
    plt.rcParams["figure.figsize"] = (15, 6)
    # Plot
    data.plot(ax=ax, x=x, y=y, style=style, grid=True, xticks=np.arange(0, data[x].shape[0], step=1), fontsize=20)
    # Set labels
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    # Set ticks
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # Set title
    ax.set_title(title, fontsize=28)
    # Set legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_weekly_analysis(df_data, name, k, path):
    """Create the report.

    Parameters
    ----------
    df_data: DataFrame
        The DataFrame of data
    name: str
        The name
    k: int
        The number of top predictions
    path: str, optional
        The path used to save the output
    """

    # Calculate true positives
    try:
        df_data['TP'] = df_data.apply(lambda row: np.sum(row.y_test.to_numpy()[row.y_prob.argsort()[::-1][0:k]]),
                                      axis=1)
    except:
        df_data['TP'] = df_data.apply(lambda row: np.sum(row.y_test[row.y_prob.argsort()[::-1][0:k]]), axis=1)
    # Create pdf
    with PdfPages(path+name+"top "+str(k)+'.pdf') as pdf:
        # Init subplots
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        ax2 = ax1.twinx()
        # Plot
        df_data['time_diff'] = range(0, df_data.shape[0])
        plot_pandas(ax1, df_data,
                    x='time_diff',
                    y=['# of escalation flags', 'TP'.format(k)],
                    style=['bo--', 'yx--'],
                    xlabel="t - to",
                    title='{} | Top{}'.format(name, k))
        plot_pandas(ax2, df_data,
                    x='time_diff',
                    y=['# of samples'],
                    style=['go--'])
        # Print # of samples and # of escalation flags
        width = 0.0
        # set text
        ax1.text(width, 0.9*df_data['# of escalation flags'].max(),
                 'Total # of samples = {}'.format(df_data['# of samples'].sum()), fontsize=20)
        ax1.text(width, 0.8*df_data['# of escalation flags'].max(),
                 'Total # of escalation flags = {}'.format(int(df_data['# of escalation flags'].sum())), fontsize=20)
        # Print num_of_escalation_flags and ratio
        num_of_escalation_flags = df_data['TP'].sum()
        ratio = round(100 * num_of_escalation_flags / df_data['# of escalation flags'].sum(), 2)
        ax1.text(width, 0.7*df_data['# of escalation flags'].max(),
                 'TP = {}'.format(num_of_escalation_flags), fontsize=20)
        ax1.text(width, 0.6*df_data['# of escalation flags'].max(),
                 'Recall = {}%'.format(ratio), fontsize=20)
        # Set legends
        ax1.legend(bbox_to_anchor=(-0.05, 1), loc='upper right', borderaxespad=0., fontsize=18)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=18)
        plt.show()
        # Save
        pdf.savefig(fig, bbox_inches="tight")


def extract_feature_importances(df_data, feature_matrix, classifier_dict, config, num_not_feat_cols=3):
    """Extract feature importances from classifier DataFrame.

    Parameters
    ----------
    df_data: DataFrame
        The DataFrame of data
    feature_matrix: DataFrame
        Feature matrix
    classifier_dict: dict
        The classifier dictionary
    config: dict
        The dict of the configurations
    num_not_feat_cols: int, optional
        The number of cols from feature_matrix which are not features. Default: 3

    Return
    ------
    np.ndarray
        The array of the feature importances (num_weeks, num_feat)
    """

    if config['late_fusion_flag']:
        feature_importances = []
        # Iterate through weeks
        for w in range(df_data.shape[0]):
            # Load classifier
            clf = pickle.loads(df_data['model'][w])
            # Extract coefs from meta classifier
            coef_1 = clf.meta_clf_.coef_[0][1] / (clf.meta_clf_.coef_[0][1] + clf.meta_clf_.coef_[0][3])
            coef_2 = clf.meta_clf_.coef_[0][3] / (clf.meta_clf_.coef_[0][1] + clf.meta_clf_.coef_[0][3])
            # Extract feature importances from base classifiers
            feat_1 = dict(zip(clf.clfs_[0].steps[0][1].cols, clf.clfs_[0].steps[1][1].feature_importances_ * coef_1))
            feat_2 = dict(zip(clf.clfs_[1].steps[0][1].cols, clf.clfs_[1].steps[1][1].feature_importances_ * coef_2))
            feature_importance = np.zeros(len(feature_matrix.columns) - num_not_feat_cols)
            # Merge feature importances
            for i in range(feature_importance.shape[0]):
                if i in feat_1.keys():
                    feature_importance[i] += feat_1[i]
                if i in feat_2.keys():
                    feature_importance[i] += feat_2[i]
            # Append feature_importances
            feature_importances.append(feature_importance)
        # Convert to array
        # Feature_importances (num_weeks, num_feat)
        feature_importances = np.array(feature_importances)
    else:
        metric = 'model'
        # Feature_importances (num_weeks, num_feat)
        feature_importances = np.array(
            [np.array(pickle.loads(classifier_dict[key][metric]).feature_importances_) for key in
             classifier_dict.keys()])
    return feature_importances


def aggregate_feature_importances(feature_importances, feature_matrix,
                                  list_not_feat_cols=['pred_time', 'escalation_flag', 'customer']):
    """Aggregate feature importances.

    Parameters
    ----------
    feature_importances: np.ndarray
        The feature_importances (num_weeks, num_feat)
    feature_matrix: DataFrame
        Feature matrix
    list_not_feat_cols: list, optional
        The list of cols from feature_matrix which are not features

    Return
    ------
    np.ndarray
        The array of the aggregated feature importances (num_feat per week, num_weeks)
    """

    # Feature importances aggregated over weeks
    # features (num_feat)
    features = feature_matrix.drop(columns=list_not_feat_cols).columns
    # Feature importances aggregated over the same feature type
    # uniq_features (num_unique_feat)
    uniq_features = np.unique([feat[:-5] for feat in features])
    # Aggregate feature_importances
    feature_importances_over_weeks = []
    for feat in uniq_features:
        feature_importances_over_weeks.append(
            np.mean([feature_importances[:, i] for i in range(len(features)) if features[i][:-5] == feat], axis=0))
    # feature_importances_over_weeks (num_feat per week, num_weeks)
    feature_importances_over_weeks = np.array(feature_importances_over_weeks)
    return feature_importances_over_weeks


def plot_feature_importances(importances, features, title, path):
    """Plot the feature importances.

    Parameters
    ----------
    importances: np.ndarray
        A array of feature importances (num_feat)
    features: list
        A list of feature names (num_feat)
    title: str
        The title
    path: str
        The path used to save plots
    """

    # Set the style
    sns.set()
    # Set size
    plt.rcParams["figure.figsize"] = (15, 6)
    # Plot the bubble plot
    plt.title(title, fontsize=30)
    # Extract weeks and features
    # weeks len(weeks)=num_feat
    weeks = [int(feat[-2:]) + 1 for feat in features]
    # features len(weeks)=num_feat
    features = [feat[:-5] for feat in features]
    # Create DataFrame with weeks, features and importances
    df = pd.DataFrame(list(zip(weeks, features, importances)), columns=['week', 'feature', 'Importances'])
    # Plot
    sns.scatterplot(x="week", y="feature", hue="Importances", size="Importances", sizes=(50, 200), data=df)
    # Set ticks
    plt.xticks(range(1, 11), fontsize=20)
    # Set labels
    plt.ylabel('', fontsize=24)
    plt.xlabel('Week', fontsize=24)
    # Set legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=20)
    # Save
    plt.savefig(path + "Feature Importances Bubble.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_aggregated_over_weeks(df_data, feature_importances_over_weeks, uniq_features, path):
    """ Plot aggregated feature importances.

    Parameters
    ----------
    df_data: DataFrame
        The classifier DataFrame
    feature_importances_over_weeks: np.ndarray
        The feature importances aggregated over weeks (num_feat per week, num_weeks)
    uniq_features: np.ndarray
        The unique features (num_feat per week)
    path: str
        The path used to save
    """

    # Define x tickes
    x_ticks = range(0, df_data.shape[0])
    # Plot
    ax = sns.heatmap(feature_importances_over_weeks, yticklabels=uniq_features,
                     xticklabels=x_ticks)
    # Set label
    ax.set_xlabel("t - to", fontsize=24)
    # Set title
    ax.axes.set_title("Feature importances", fontsize=30)
    # Set ticks
    plt.xticks(fontsize=12)
    # Save
    plt.savefig(path + "Feature Importances.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def calculate_weight_ratio(df_data):
    """Calculate weight ratio between enterprise and log classifier in case of late fusion.

    Parameters
    ----------
    df_data: DataFrame
        The classifier DataFrame

    Return
    ------
    list
        A list of ratio (num_weeks)
    """

    weights = []
    for w in range(df_data.shape[0]):
        # Load classifier
        clf = pickle.loads(df_data['model'][w])
        # Extract coefs from meta classifier
        coef_1 = clf.meta_clf_.coef_[0][1] / (clf.meta_clf_.coef_[0][1] + clf.meta_clf_.coef_[0][3])
        coef_2 = clf.meta_clf_.coef_[0][3] / (clf.meta_clf_.coef_[0][1] + clf.meta_clf_.coef_[0][3])
        weights.append(coef_1/coef_2)
    return weights


def plot_ratio(df_data, path):
    """Plot ratio between enterprise and log classifier in case of late fusion.

    Parameters
    ----------
    df_data: DataFrame
        The classifier DataFrame
    path: str
        The path used to save
    """

    # Plot
    plt.plot(calculate_weight_ratio(df_data), '*--')
    # Set title
    plt.title("Ratio between enteprise base classifier and log base classifier", fontsize=28)
    # Set labels
    plt.ylabel("Ratio", fontsize=20)
    plt.xlabel("t - to", fontsize=20)
    # Saves
    plt.savefig(path + "Ratio.pdf", format="pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    pass
