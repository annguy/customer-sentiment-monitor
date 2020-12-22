import torch
import torch.nn as nn
import numpy as np
import pickle
import warnings
import time
import os
import optuna
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from time import sleep
from src.general_helper_functions import add_week_to_cwdate, calculate_recall_at_k,\
                                          create_model_info, cal_week_gen, filter_and_split_feature_matrix_by_cal_week,\
                                          update_log


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def predict(model, dataset, loss):
    """Predict probabilities.

    Parameters
    ----------
    model: torch.nn.Module or Model
        The model/architecture
    dataset: torch.utils.data.Dataset
        The dataset
    loss:
        The loss function

    Returns
    -------
    List
        The list of target values
    List
        The list of predicted probabilities
    List
        The list of predicted values
    Float
        The loss value between predicted and target values
    """

    # Set seed
    seed_everything()
    # Use GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    # Set evaluation mode
    model.eval()
    with torch.no_grad():
        y, y_prob = None, None
        # Iterate through batches
        for features, labels in dataset:
            # labels (batch_size)
            # features (batch_size, seq_len, input_size)
            labels = labels.type('torch.FloatTensor').to(model.device)
            batch_size = len(features)
            # If batch_size less than model.batch_size add zeros
            if batch_size != model.batch_size:
                features = torch.cat((features, torch.zeros(model.batch_size - batch_size, 10, 32).type(torch.float64)))
            # output (model.batch_size)
            output = model(features)
            # output (batch_size)
            output = output[:batch_size]
            # Concatenate target and prediction probabilities
            y, y_prob = (labels, output) if y is None else (torch.cat((y, labels)), torch.cat((y_prob, output)))
    # y_pred (num_of_samples)
    y_pred = torch.tensor(torch.where(y_prob > 0.5, 1, 0))
    return y, y_prob, y_pred, loss(y_prob, y.to(model.device))


def train(model, dataset, loss, optimizer):
    """Train the torch model.

    Parameters
    ----------
    model: torch.nn.Module or Model
        The model/architecture
    dataset: torch.utils.data.Dataset
        The dataset
    loss:
        The loss function
    optimizer: torch.optima
        The optimizer

    Return
    ------
    Float
        The loss value
    """

    # Use GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    # Set the training mode
    model.train()
    # Init
    y, y_prob = None, None
    # Iterate through batches
    for features, labels in dataset:
        # Forward Pass
        # features (batch_size, seq_len, input_size), labels (batch_size)
        # output (batch_size)
        output = model(features)
        # Calculate loss value
        loss_out = loss(output, labels.type('torch.FloatTensor').to(model.device))
        # Backward pass
        optimizer.zero_grad()
        # Clears the gradients of all optimized torch.Tensor's.
        loss_out.backward()
        # Performs a single optimization step - updating parameters(weights and bias)
        optimizer.step()
        # Concatenate targets and predictions
        y, y_prob = (labels, output) if y is None else (torch.cat((y, labels)), torch.cat((y_prob, output)))
    # If dataset is empty return 0, else return loss function through all data set
    if len(dataset) == 0:
        return None
    else:
        return loss(y_prob, y.type('torch.FloatTensor').to(model.device))


def seed_everything(seed=42):
    """Seed.

    Parameter
    ---------
    seed: int
        The seed
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_list_dataset(dataset):
    """Convert data set array into list of arrays, where each list element
    belongs to the unique pred_time.

    Parameters
    ----------
    dataset: pd.DataFrame (num_sampled, )
        The DataFrame with feature matrix containing 4 columns:
        customer, pred_time, feat and label.

    Returns
    -------
    list
        A list of feature arrays len()=num_unique_pred_time
    list
        A list of label arrays len()=num_unique_pred_time
    """

    # len(x_train) = num_train_pred_time, shape(x_train[0]) == (num_samples, seq_len, num_features)
    # len(y_train) = num_train_pred_time, shape(y_train[0]) == (num_samples)
    list_x, list_y = [], []
    # Iterate through pred_time
    for pred_time in dataset.pred_time.unique():
        # Slice data
        dataset_slice = dataset[dataset.pred_time == pred_time]
        # x (num_samples per pred_time, seq_len, num_features)
        # y (num_samples per pred_time)
        x, y = np.array(dataset_slice.feat.tolist()), np.array(dataset_slice.label.tolist())
        # Append x and y
        list_x.append(x), list_y.append(y)
    return list_x, list_y


def output_report(dataset, clf, loss_function=torch.nn.BCELoss(), test_flag=False):
    """Output/print the confusion matrix, average recalls at k and positions of escalation_flag.

    Parameters
    ----------
    dataset: torch.utils.data.DataLoader
        The DataLoader
    clf: Model or nn.
        The classifier
    loss_function:
        The loss function
    test_flag: bool, optional
        Output positions of escalation flags if the test set is used. Default: False
    """

    if isinstance(dataset, list):
        recall_at_k, avg_recall, _ = calculate_recall_at_k_time_series(dataset, clf, loss_function)
        y, y_prob, y_pred = None, None, None
        for subset in dataset:
            y_set, y_prob_set, y_prob_set, _ = predict(clf, subset, loss_function)
            y, y_prob, y_pred = (y_set, y_prob_set, y_prob_set) \
                if y is None else (torch.cat((y, y_set)),
                                   torch.cat((y_prob, y_prob_set)),
                                   torch.cat((y_pred, y_prob_set)))
    else:
        # Make prediction on train set
        y, y_prob, y_pred, _ = predict(clf, dataset, loss_function)
        # Calculate the recalls_At_l
        _, recall_at_k, avg_recall, _, _ = calculate_recall_at_k(y_prob.cpu().data.numpy(),
                                                                 y.cpu().data.numpy(), k_max=100)
    # Output confusion_matrix
    print("Confusion matrix: \n", confusion_matrix(y.cpu().data.numpy(),
                                                   y_pred.cpu().data.numpy()))
    # Output recall_at_k
    print("Recall at {}, {}, {}, {}, {}: ".format(5, 10, 20, 50, 100), end=' ')
    for top in [5, 10, 20, 50, 100]:
        if top == 100:
            print(str(round(float(recall_at_k[top - 1]), 2)) + " accordingly")
        else:
            print("{}, ".format(round(float(recall_at_k[top - 1]), 2)), end='')
    print("Average recalls over 100: ", round(float(avg_recall), 2))
    if test_flag:
        if isinstance(dataset, list):
            recall_at_k, _, _ = calculate_recall_at_k_time_series(dataset, clf, loss_function)
        else:
            _, recall_at_k, _, _, _ = calculate_recall_at_k(y_prob.cpu().data.numpy(), y.cpu().data.numpy(),
                                                            k_max=y.cpu().data.numpy().shape[0])
        print(
            "Positions of escalation flags: ", ([1] if recall_at_k[0] != 0 else []) +
            [i + 1 for i in range(1, len(recall_at_k)) if recall_at_k[i] != recall_at_k[i - 1]])


def create_dataloader(features, labels, batch_size=128, drop_last=False, shuffle=False):
    """Create data loader.

    Parameters
    ----------
    features: np.ndarray
        The target values (num_time_steps, num_samples, num_features)
    labels: np.ndarray
        The predictive values (num_samples, )
    batch_size: int, optional
        The batch size. Default: 128
    drop_last: bool, optional
        If the last batch is dropped. Default: False
    shuffle: bool, optional
        If the data are shuffled. Default: False

    Return
    ------
    torch.utils.data.Dataset
        The dataset
    """

    # Convert torch to numpy
    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels).type('torch.FloatTensor')
    # Create data set
    dataset = torch.utils.data.TensorDataset(features, labels)
    # Create data loader
    dataset = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    return dataset


def suggest_parameters(trial, config):
    """Init/suggest parameters for optuna.

    Parameters
    ----------
    trial: optuna.
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
        elif parameters[key][0] == 'fixed':
            optuna_parameters[key] = parameters[key][1]
    return optuna_parameters


def objective(trial, x_train, y_train, x_val, y_val, config, data_path):
    """Implement objective function for the Optuna.

    Parameters
    ----------
    trial: optuna.trial
    x_train: list
        A list of the train features. len(x_train) = num_train_pred_time,
        shape(x_train[0]) == (num_samples, seq_len, num_features)
    y_train: list
        A list of the train labels. len(y_train) = num_train_pred_time,
        shape(y_train[0]) == (num_samples)
    x_val: list
        A list of the validation features. len(x_val) = num_val_pred_time,
        shape(x_val[0]) == (num_samples, seq_len, num_features)
    y_val: list
        A list of the validation labels. len(y_val) = num_val_pred_time,
        shape(y_val[0]) == (num_samples)
    config: dict
        The configurations
    data_path: str
        A path used for saving trials

    Return
    ------
    Float
        The validation metric
    """

    # Init params for trial
    params = suggest_parameters(trial, config)
    # Create DataLoader
    # Flat the list of arrays for training
    # x_train_flat (num_samples, seq_len, num_features)
    x_train_flat = np.array([item for sublist in x_train for item in sublist])
    # y_train_flat (num_samples)
    y_train_flat = np.array([item for sublist in y_train for item in sublist])
    # Create customer dataset
    trainset_flat = Customer_Dataset(x_train_flat,
                                     y_train_flat,
                                     sampling_strategy=params['sampling_strategy'],
                                     sampling_ratio=params['ratio'])
    # # Create FastDataLoader using trainset_flat
    trainset_flat = FastDataLoader(dataset=trainset_flat,
                                   batch_size=params['batch_size'],
                                   drop_last=True,
                                   shuffle=True)

    # trainset (len(trainset) = num_train_pred_time)
    trainset = [create_dataloader(x, y, batch_size=params['batch_size']) for x, y in zip(x_train, y_train)]
    # valset (len(valset) = num_val_pred_time)
    valset = [create_dataloader(x, y, batch_size=params['batch_size']) for x, y in zip(x_val, y_val)]
    # input_size equal to number of features
    input_size = x_train_flat.shape[2]
    # Set the model
    model = Model(input_size, params['hidden_dim'], params['lstm_layer'], params['batch_size'],
                  params['dropout_prob'], bidirectional=params['lstm_bidirectional'])
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=params['weight_decay'], lr=params['learning_rate'])
    # Set loss function
    loss_function = torch.nn.BCELoss()
    # Init metrics
    best_avg_recall_val, best_model, count_epochs = None, None, 0
    avg_recalls_train, train_losses, avg_recalls_val, val_losses = [], [], [], []
    # Iterate through epochs
    # Use sleep to correct visualization of tqdm bar
    sleep(0.1)
    for epoch in tqdm(range(params['num_epochs'])):
        # Train
        train(model, trainset_flat, loss_function, optimizer)
        _, avg_recall_train, train_loss = calculate_recall_at_k_time_series(trainset, model, loss_function, k_max=100)
        # Validation
        _, avg_recall_val, val_loss = calculate_recall_at_k_time_series(valset, model, loss_function, k_max=100)
        # Append metrics
        avg_recalls_train.append(avg_recall_train), train_losses.append(train_loss)
        avg_recalls_val.append(avg_recall_val), val_losses.append(val_loss)
        # Update metrics if the current trial is the best
        if (best_avg_recall_val is None) or (avg_recall_val > best_avg_recall_val):
            best_model = copy.deepcopy(model)
            best_avg_recall_val = avg_recall_val
            count_epochs = 0
        count_epochs += 1
        # Terminate if early stopping conditions satisfy
        if params['early_stopping'] and count_epochs > config['patience']:
            break
        # Report intermediate objective value.
        trial.report(avg_recall_val, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            break
    sleep(0.1)
    callback = {'avg_recalls_train': avg_recalls_train, 'train_losses':  train_losses,
                'avg_recalls_val': avg_recalls_val, 'val_losses': val_losses}
    # Save trial
    with open(data_path + 'interim/trial_{}.pickle'.format(trial.number), 'wb') as f:
        pickle.dump((best_model, callback), f)
    return best_avg_recall_val


def calculate_recall_at_k_time_series(dataset, model, loss_function, k_max=100):
    """Calculate recall at k for time series data. For each pred_time:
    use calculate_recall_at_k. Then combine outputs from calculate_recall_at_k.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        The dataset
    model: torch.nn.Module or Model
        The model
    loss_function:
        The loss function
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
    """

    # Init arrays
    # tps (k_max), flags (k_max)
    tps, flags = np.zeros(k_max), np.zeros(k_max)
    y, y_prob = None, None
    # Iterate through pred_time
    for subset in dataset:
        # Predict
        # y_set (num_samples per pred_time)
        # y_prob_set (num_samples per pred_time)
        y_set, y_prob_set, _, _ = predict(model, subset, loss_function)
        # Calculate recall at k for pred_time
        # len(num_of_tp) = k_max
        # len(num_of_flag) = k_max
        _, _, _, num_of_tp, num_of_flag = calculate_recall_at_k(y_prob_set.cpu().data.numpy(),
                                                                y_set.cpu().data.numpy(), k_max=k_max)
        # Update tps and escalation flags
        # tps (k_max)
        tps += np.array(num_of_tp)
        # flags (k_max)
        flags += np.array(num_of_flag)
        # Concatenate y and y_prob
        y, y_prob = (y_set, y_prob_set) if y is None else (torch.cat((y, y_set)), torch.cat((y_prob, y_prob_set)))
    return np.nan_to_num(tps/flags), np.mean(np.nan_to_num(tps/flags)), loss_function(y_prob, y.to(model.device)).item()


def tpe_sampler_search(x_train, y_train,
                       x_val, y_val,
                       config, data_path):
    """Use optuna.TPESampler to tune hyperparameters.

    Parameters
    ----------
    x_train: list
        A list of the train features. len(x_train) = num_train_pred_time,
        shape(x_train[0]) == (num_samples, seq_len, num_features)
    y_train: list
        A list of the train labels. len(y_train) = num_train_pred_time,
        shape(y_train[0]) == (num_samples)
    x_val: list
        A list of the validation features. len(x_val) = num_val_pred_time,
        shape(x_val[0]) == (num_samples, seq_len, num_features)
    y_val: list
        A list of the validation labels. len(y_val) = num_val_pred_time,
        shape(y_val[0]) == (num_samples)
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
    # Create study and set MedianPruner
    study = optuna.create_study(sampler=sampler, direction='maximize',
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=config['n_startup_trials'],
                                                                   n_warmup_steps=config['n_warmup_steps'],
                                                                   interval_steps=config['interval_steps']))
    # Disable output
    optuna.logging.disable_default_handler()
    # Optimize
    study.optimize(lambda trial: objective(trial, x_train, y_train,
                                           x_val, y_val, config, data_path),
                   n_trials=n_trials)
    # Init model with best parameters
    print("Best trial: ", study.best_trial.number)
    print("Best parameters: ", study.best_params)
    # Load the best trial.
    with open(data_path + 'interim/trial_{}.pickle'.format(study.best_trial.number), 'rb') as f:
        model, callback = pickle.load(f)
    # Delete all trials
    for trial_num in range(config['number_of_trials']):
        if os.path.exists(data_path + 'interim/trial_{}.pickle'.format(trial_num)):
            os.remove(data_path + 'interim/trial_{}.pickle'.format(trial_num))
    # Create DataLoaders
    # trainset (len(trainset)=pred_time)
    trainset = [create_dataloader(x, y, batch_size=model.batch_size) for x, y in zip(x_train, y_train)]
    # valset (len(valset)=pred_time)
    valset = [create_dataloader(x, y, batch_size=model.batch_size) for x, y in zip(x_val, y_val)]
    print("***Train***")
    output_report(trainset, model)
    print("***Validation***")
    output_report(valset, model)
    callback['trials_dataframe'] = study.trials_dataframe()
    return model, callback


def analyze_weekly(feature_matrix, config, data_path):
    """Train a classifier using all data previous of week N for training and then
       predict escalation flag for week N+1 and N+2.
       Assuming that week N is the last week for which all data is available.

    Parameters
    ----------
    feature_matrix: pd.DataFrame (num_samples, 4)
        The DataFrame with feature matrix containing 4 columns:
        customer, pred_time, feat and label.
    config: dict
        The configurations
    data_path: str
        The path used to save and load data

    Return
    -------
    classifier_dict : dict{week : (model_info)}
        Dictionary containing the trained models for each week
    """

    if torch.cuda.is_available():
        print('Running on GPU')
    else:
        print('Running on CPU')
    # Print/output model, fusion and feature types
    if config['model_name'] == 'LSTM':
        print("LSTM is used")
    if config['late_fusion_flag']:
        print("Late fusion is used")
    else:
        print("Early fusion is used")
    if config['feature_type'] == 'both_feature_types':
        print("Both enterprise and log features are used")
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
    for pred_time in cal_week_gen(first_pred_time, last_prediction_time):
        print("*************************************************************")
        print("Prediction time: {}".format(pred_time))
        # Log start time to calculate the elapsed time later
        time_start = time.time()
        # Calculate the last used time for train_val set
        # Format: YYYY-WW (according to ISO8601)
        last_time = add_week_to_cwdate(pred_time, weeks=-config['lookahead_window_len'] -
                                                         config['fixed_window_in_weeks']) \
            if config['fixed_window_in_weeks'] != -1 \
            else add_week_to_cwdate(feature_matrix.pred_time.min(), weeks=-1)
        # Divide into train_val and test DataFrames
        feature_matrix_train_val, feature_matrix_test = filter_and_split_feature_matrix_by_cal_week(
            feature_matrix[feature_matrix.pred_time > last_time], pred_time, config)
        # Split into train and validation set with (lookahead_window_len week-1) gap
        # Format: YYYY-WW (according to ISO8601)
        last_val_week = feature_matrix_train_val.pred_time.max()
        # Format: YYYY-WW (according to ISO8601)
        first_val_week = add_week_to_cwdate(last_val_week, weeks=-config['num_of_val_weeks'] + 1)
        # Format: YYYY-WW (according to ISO8601)
        last_train_week = add_week_to_cwdate(first_val_week, weeks=-config['lookahead_window_len'])
        # Get train set
        train_data = feature_matrix_train_val[feature_matrix_train_val.pred_time <= last_train_week]
        # Get validation set
        val_data = feature_matrix_train_val[feature_matrix_train_val.pred_time >= first_val_week]
        print("Train set from {} to {}".format(train_data.pred_time.min(), train_data.pred_time.max()))
        print("Validation set from {} to {}".format(val_data.pred_time.min(), val_data.pred_time.max()))
        # Init scaler
        sc = StandardScaler()
        # Standardize train
        # x_train (num_train_pred_time, num_samples, seq_len, num_features)
        x_train = np.array(train_data.feat.tolist())
        sc.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
        # Convert array into list of arrays
        # len(x_train) = num_train_pred_time, shape(x_train[0]) == (num_samples, seq_len, num_features)
        # len(y_train) = num_train_pred_time, shape(y_train[0]) == (num_samples)
        x_train, y_train = create_list_dataset(train_data)
        # Standardize x
        x_train = [sc.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape) for x in x_train]
        # Convert array into list of arrays
        # len(x_val) = num_val_pred_time, shape(x_val[0]) == (num_samples, seq_len, num_features)
        # len(y_val) = num_val_pred_time, shape(y_val[0]) == (num_samples)
        x_val, y_val = create_list_dataset(val_data)
        # Standardize x
        x_val = [sc.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape) for x in x_val]
        # Can avoid seq_len, because num of timestamp is equal to 1
        # x_test (num_samples, seq_len, num_features)
        # y_test (num_samples)
        x_test, y_test = np.array(feature_matrix_test.feat.tolist()), np.array(feature_matrix_test.label.tolist())
        # Standardize test
        x_test = sc.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
        # TPESampler search = Tune hyperparameter
        classifier, info = tpe_sampler_search(x_train, y_train, x_val, y_val, config, data_path)
        # Train classifier
        loss_function = torch.nn.BCELoss()
        best_trial = info['trials_dataframe'].value.argmax()
        batch_size = info['trials_dataframe'].loc[best_trial, :].params_batch_size
        # Create test loader
        testset = create_dataloader(x_test, y_test, batch_size=int(batch_size))
        # Make prediction on test set
        y_test, y_prob, y_pred, _ = predict(classifier, testset, loss_function)
        # Create model info
        model_info = create_model_info(classifier,
                                       config, info,
                                       y_test.cpu().data.numpy(),
                                       y_pred.cpu().data.numpy(),
                                       y_prob.cpu().data.numpy())
        classifier_dict[pred_time] = model_info
        print("***Test***")
        output_report(testset, classifier, test_flag=True)
        # Update feature_matrix and (classifier_dict, collective_pred_mat)
        update_log(data_path + '/interim/results_{}.pickle', classifier_dict, pred_time)
        time_diff = time.time() - time_start
        print("Time elapsed: {} minutes".format(round(time_diff / 60, 2)))
        print("*************************************************************")
    return classifier_dict


class Customer_Dataset(Dataset):
    """ Define the customer map-style dataset by defining __getitem__ and __len__ methods.
    Oversample or undersample dataset once per epoch.

    Attributes
    ----------
    features: numpy.ndarray (num_samples, seq_len, num_features)
            The features in the input x
    label: numpy.ndarray (num_samples, )
        The labels
    sampling_strategy: str
        The sampling strategy: None, 'under' or 'over'.
    sampling_ratio: float
        The sampling ratio. The ratio is equal to the number of samples
        in the minority class over the number of samples in the majority
        class after resampling. sampling_ratio varies (0, 1].

    Methods
    -------
    new_epoch()
        Sample and shuffle dataset each epoch
    perform_oversampling(sampling_ratio)
        Perform oversampling
    perform_undersampling(sampling_ratio)
        Perform undersampling
    shuffle()
        Shuffle sampled dataset
    """

    def __init__(self,
                 features,
                 label,
                 sampling_strategy=None,
                 sampling_ratio=1.):
        """
        Parameters
        ----------
        features: numpy.ndarray (num_samples, seq_len, num_features)
            The features in the input x
        label: numpy.ndarray (num_samples, )
            The labels
        sampling_strategy: str, optional
            The sampling strategy: None, 'under' or 'over'.  Default: None
        sampling_ratio: float, optional
            The sampling ratio. The ratio is equal to the number of samples
            in the minority class over the number of samples in the majority
            class after resampling. Sampling_ratio varies (0, 1]. Default: 1.0
        """

        self.features = features
        self.label = label
        self.sampling_strategy = sampling_strategy
        self.sampling_ratio = sampling_ratio
        # Sample
        if self.sampling_strategy == 'under':
            self.perform_undersampling(self.sampling_ratio)
        elif self.sampling_strategy == 'over':
            self.perform_oversampling(self.sampling_ratio)
        elif self.sampling_strategy is None:
            self.sampled_features, self.sampled_label = pd.DataFrame(pd.Series(list(self.features)),
                                                                     columns=['features']), \
                                                        self.label
        elif self.sampling_strategy is not None:
            print('ERROR - got unknown sampling type: ' + str(self.sampling_strategy))
        # Shuffle
        self.shuffle()

    def new_epoch(self):
        """Sample and shuffle dataset each epoch.
        """

        # Sample
        if self.sampling_strategy == 'under':
            self.perform_undersampling(self.sampling_ratio)
        elif self.sampling_strategy == 'over':
            self.perform_oversampling(self.sampling_ratio)
        else:
            self.sampled_features, self.sampled_label = pd.DataFrame(pd.Series(list(self.features)),
                                                                     columns=['features']), self.label
        # Shuffle sampled dataset
        self.shuffle()

    def perform_oversampling(self, sampling_ratio):
        """Oversample dataset with sampling_ratio.

        Parameter
        ---------
        sampling_ratio: float
            The sampling ratio. The ratio is equal to the number of samples
            in the minority class over the number of samples in the majority
            class after resampling. Sampling_ratio varies (0, 1].
        """

        # Init sampler
        sampler = RandomOverSampler(sampling_strategy=sampling_ratio, random_state=0)
        # Oversample
        sampled_features, sampled_label = sampler.fit_resample(pd.DataFrame(pd.Series(list(self.features)),
                                                                            columns=['features']),
                                                               self.label)
        # Assign to the class attributes
        self.sampled_features, self.sampled_label = sampled_features.copy(), sampled_label.copy()

    def perform_undersampling(self, sampling_ratio):
        """Undersample dataset with sampling_ratio.

        Parameter
        ---------
        sampling_ratio: float
            The sampling ratio. The ratio is equal to the number of samples
            in the minority class over the number of samples in the majority
            class after resampling. Sampling_ratio varies (0, 1].
        """

        # Init sampler
        sampler = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=0)
        # Undersample
        sampled_features, sampled_label = sampler.fit_resample(pd.DataFrame(pd.Series(list(self.features)),
                                                                            columns=['features']),
                                                               self.label)
        # Assign to the class attributes
        self.sampled_features, self.sampled_label = sampled_features.copy(), sampled_label.copy()

    def shuffle(self):
        """ Shuffle dataset.
        """

        # Combine features and label into one DataFrame
        self.sampled_features['label'] = self.sampled_label
        # Shuffle
        self.sampled_features = self.sampled_features.sample(frac=1.).reset_index(drop=True)
        # Split DataFrame into self.sampled_features.label and self.sampled_features
        self.sampled_label, self.sampled_features = self.sampled_features.label, self.sampled_features.drop(
            columns=['label'])

    def __len__(self):
        return self.sampled_features.shape[0]

    def __getitem__(self, idx):
        # Convert idx to list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Slice the labels and features based on idx
        label = self.sampled_label.iloc[idx]
        feature = self.sampled_features.iloc[idx].features
        return feature, label


class _RepeatSampler(object):
    """ Sampler that repeats forever. The wrapper class to convert
    map-style into iterable-style dataset by defining __iter__ method.

    Attribute
    ----------
    sampler: torch.utils.data.Sampler
        The sampler
    """

    def __init__(self, sampler):
        """
        Attribute
        ----------
        sampler: torch.utils.data.Sampler
            The sampler
        """
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    """ A customer loader to use iterable-style dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

    def new_epoch(self):
        self.dataset.new_epoch()


class Model(nn.Module):
    """ A class used to define the model.

    Attributes
    ----------
    input_size: int
        The number of expected features in the input x
    hidden_size: int
        The number of features in the hidden state h
    num_layers: int
        Number of recurrent layers
    batch_size: int
        Batch size
    dropout_prob: float
        The probability of an element to be zeroed
    bidirectional: bool
        If True, becomes a bidirectional LSTM
    num_directions: int
        The number of lstm directions
    lstm: torch.nn.LSTM
        The LSTM layer
    dropout: torch.nn.Dropout
        The dropout layer
    batch_norm_1d: torch.nn.BatchNorm1d
        The batch normalization layer
    linear: torch.nn.Linear
        The linear layer
    softmax: torch.nn.Softmax
        The softmax layer
    h_n_and_c_n: tuple
        The hidden and cell state. len(h_n_and_c_n) = 2

    Methods
    -------
    init_hidden()
        Initialize hidden and cell states
    forward(input)
        Forward pass
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 batch_size, dropout_prob, bidirectional=False):
        """
        Parameters
        ----------
        input_size: int
            The number of expected features in the input x
        hidden_size: int
            The number of features in the hidden state h
        num_layers: int
            Number of recurrent layers
        batch_size: int
            Batch size
        dropout_prob: float
             The probability of an element to be zeroed
        bidirectional: bool, optional
            If True, becomes a bidirectional LSTM. Default: False
        """

        super(Model, self).__init__()
        # Utilize CPU or GPU
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        # Attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # Define LSTM layer
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional)
        # Define dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)
        # Define batch normalization layer
        self.batch_norm_1d = nn.BatchNorm1d(self.num_directions * self.hidden_size)
        # Define linear layer
        self.linear = nn.Linear(self.hidden_size*self.num_directions, 2)
        # Define softmax layer
        self.softmax = nn.Softmax(dim=1)
        # Init hidden and cell states
        self.h_n_and_c_n = None

    def init_hidden(self):
        """Initialize hidden and cell states.
        """

        # Init hidden and cell states
        self.h_n_and_c_n = (torch.zeros(self.num_layers*self.num_directions,
                                        self.batch_size,
                                        self.hidden_size,
                                        device=self.device),
                            torch.zeros(self.num_layers*self.num_directions,
                                        self.batch_size,
                                        self.hidden_size,
                                        device=self.device))

    def forward(self, input_seq):
        """Forward pass.

        Parameters
        ----------
        input_seq: torch.Tensor (batch_size, seq_len, input_size)
            The input tensor

        Returns
        ----------
        torch.Tensor
            The predictions
        """

        # Init hidden and cell states
        self.init_hidden()
        # LSTM layer
        # Output (batch_size, seq_len, num_directions * hidden_size)
        # h_n_and_c_n ((batch, num_layers * num_directions, hidden_size),
        #              (batch, num_layers * num_directions, hidden_size))
        output, self.h_n_and_c_n = self.lstm(input_seq.type('torch.FloatTensor').to(self.device), self.h_n_and_c_n)
        # Choose the last output
        # Output (batch_size, num_directions * hidden_size)
        output = output[:, -1, :]
        # BatchNormalization
        # Output (batch_size, num_directions * hidden_size)
        self.batch_norm_1d(output)
        # Dropout
        # Output (batch_size, num_directions * hidden_size)
        output = self.dropout(output)
        # Predictions
        # predictions (batch_size, 2)
        predictions = self.linear(output)
        # predictions (batch_size)
        predictions = self.softmax(predictions)[:, 0]
        return predictions


if __name__ == '__main__':
    pass
