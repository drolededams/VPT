import inspect
import re
import argparse
import pickle
import mne
import sys
import warnings

import numpy as np
import scipy as scp

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP, FilterEstimator
from mne.time_frequency import psd_multitaper
from mne.realtime import MockRtClient, RtEpochs
from mne import create_info


from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin


def describe(arg):
    frame = inspect.currentframe()
    callerframeinfo = inspect.getframeinfo(frame.f_back)
    try:
        context = inspect.getframeinfo(frame.f_back).code_context
        caller_lines = ''.join([line.strip() for line in context])
        m = re.search(r'describe\s*\((.+?)\)$', caller_lines)
        if m:
            caller_lines = m.group(1)
            position = str(callerframeinfo.filename) + "@" + str(callerframeinfo.lineno)

            # Add additional info such as array shape or string length
            additional = ''
            if hasattr(arg, "shape"):
                additional += "[shape={}]".format(arg.shape)
            elif hasattr(arg, "__len__"):  # shape includes length information
                additional += "[len={}]".format(len(arg))

            # Use str() representation if it is printable
            str_arg = str(arg)
            str_arg = str_arg if str_arg.isprintable() else repr(arg)

            print(position, "describe(" + caller_lines + ") = ", end='')
            print(arg.__class__.__name__ + "(" + str_arg + ")", additional)
        else:
            print("Describe: couldn't find caller context")

    finally:
        del frame
        del callerframeinfo


class myCSP(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_channels = X.shape[1]

        covs = np.zeros((n_classes, n_channels, n_channels))
        for class_id in classes:
            X_classed = X[y == class_id]
            X_classed = np.transpose(X_classed, [1, 0, 2])
            X_classed = X_classed.reshape(n_channels, -1)
            cov = X_classed.dot(X_classed.T) / (X_classed.shape[1])
            covs[class_id] = cov
        eigen_val, eigen_vecs = scp.linalg.eigh(covs[0], covs[0] + covs[1])
        ix = np.argsort(np.abs(eigen_val - 0.5))[::-1]
        eigen_vecs = eigen_vecs[:, ix]
        self.spatialFilters_ = eigen_vecs.T
        self.spatialFilters_ = self.spatialFilters_[:self.n_components]
        return self

    def transform(self, X):
        if X.ndim == 3:
            X_filtred = np.asarray([np.dot(self.spatialFilters_, epoch) for epoch in X])
            calc_var(X_filtred)
            X_filtred = np.log(X_filtred.var(axis=2, ddof=1)) # manual var ?
        else:
            X_filtred = np.asarray([np.dot(self.spatialFilters_, X)])
            calc_var(X_filtred)
            X_filtred = np.log(X_filtred.var(axis=2, ddof=1)) # manual var ?
        return X_filtred


def calc_var(X):
    describe(X)
    for x in X:
        describe(x)
        n_times = x.shape[1]
        mean = x.sum(axis=1) / n_times
        describe(mean)
        describe(x - mean)
    describe(X.var(axis=2, ddof=1))


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "mode",
            help="Mode : train, prediction or real time prediction",
            nargs='?',
            default="train",
            choices=["train", "predict", "rtpredict"])
    parser.add_argument(
            "-s",
            "--subject",
            help="Select a subject (1 to 109)",
            type=int,
            choices=range(1, 109),
            default=1)
    parser.add_argument(
            "-t",
            "--task",
            help="Select a task: - 1: open and close left or right fist - 2:imagine opening and closing left or right fist - 3:open and close both fists or both feet - 4:imagine opening and closing both fists or both feet",
            type=int,
            choices=range(1, 5),
            default=1)
    parser.add_argument(
            "-nt",
            "--n_task",
            help="Select the runs task",
            nargs='+',
            type=int,
            choices=range(1, 4),
            default=[1, 2, 3])
    parser.add_argument(
            "-ti",
            "--tmin",
            help="Select time in seconds before the trigger",
            type=float,
            choices=[Range(-10., 0.)],
            default=-1.)
    parser.add_argument(
            "-ta",
            "--tmax",
            help="Select time in seconds after the trigger",
            type=float,
            choices=[Range(0.1, 10.)],
            default=5.)
    parser.add_argument(
            "-lf",
            "--l_freq",
            help="Select the frequence below which to filter out of the data",
            type=float,
            choices=[Range(0, 79.)],
            default=8.)
    parser.add_argument(
            "-hf",
            "--h_freq",
            help="Select the frequence above which to filter out of the data",
            type=float,
            choices=[Range(1, 79.9)],
            default=32.)
    parser.add_argument(
            "-ci",
            "--crop_min",
            help="Start time in seconds of the croped epoch",
            type=float,
            default=0.5)
    parser.add_argument(
            "-ca",
            "--crop_max",
            help="End time in seconds of the croped epoch",
            type=float,
            default=2.5)
    parser.add_argument(
            "-i",
            "--iterations",
            help="Number of re-shuffling & splitting iterations.",
            type=check_positive,
            default=10)
    parser.add_argument(
            "-nc",
            "--n_components",
            help="Select a number of components for the model",
            type=int,
            choices=range(1, 65),
            default=4)
    parser.add_argument(
            "-pl",
            "--plot",
            help="Plot data",
            action="store_true")
    parser.add_argument(
            '-f',
            '--data_format',
            help='Select data formt',
            choices=['normal', 'psd'],
            default='normal')
    parser.add_argument(
            "-pref",
            "--pre_filter",
            help="Pre Filter for real time prediction",
            action="store_true")
    parser.add_argument(
            "-sf",
            "--saved_filter",
            help="Use saved filter",
            action="store_true")
    args = parser.parse_args()
    verif_args(args)
    return args


def verif_args(args):
    if args.l_freq >= args.h_freq:
        print("Argument Error: Low cut-off frequency must be greater than High cut-off frequency")
        sys.exit(0)
    if args.tmin > args.crop_min:
        print("Argument Error: Time before events must be lower than time minimum croped")
        sys.exit(0)
    if args.tmax < args.crop_max:
        print("Argument Error: Time after events must be greater than time maximum croped")
        sys.exit(0)
    if args.crop_max - 0.1 < args.crop_min:
        print("Argument Error: Time maximum croped must be greater than time minimum croped")
        sys.exit(0)


def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))


def load_args():
    args_filename = 'args.sav'
    try:
        loaded_args = pickle.load(open(args_filename, 'rb'))
    except Exception as e:
        print("Can't use pickle on {}".format(args_filename))
        print(e.__doc__)
        sys.exit(0)
    return loaded_args


def load_model():
    model_filename = 'model.sav'
    try:
        loaded_model = pickle.load(open(model_filename, 'rb'))
    except Exception as e:
        print("Can't use pickle on {}".format(model_filename))
        print(e.__doc__)
        sys.exit(0)
    return loaded_model


def save_model(X, y, pipeline, args):
    pipeline.fit(X, y)
    model_filename = 'model.sav'
    args_filename = 'args.sav'
    try:
        pickle.dump(pipeline, open(model_filename, 'wb'))
    except Exception as e:
        print("Can't use pickle on {}".format(model_filename))
        print(e.__doc__)
        sys.exit(0)
    try:
        pickle.dump(args, open(args_filename, 'wb'))
    except Exception as e:
        print("Can't use pickle on {}".format(args_filename))
        print(e.__doc__)
        sys.exit(0)


def list_runs(args):
    task = {
            1: '1 - Open and close left or right fist',
            2: '2 - Imagine opening and closing left or right fist',
            3: '3 - Open and close both fists or both feet',
            4: '4 - Imagine opening and closing both fists or both feet',
            }
    runs = []

    for i in args.n_task:
        runs.append(args.task + 2 + (i - 1) * 4)

    print("Subject:", args.subject)
    print("Task:", task[args.task])
    print("Runs =", runs)

    return runs


def get_data(args):
    path = '/sgoinfre/goinfre/Perso/dgameiro/TPV_database'
    runs = list_runs(args)
    raws = eegbci.load_data(args.subject, runs, path=path)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raws])
    return raw


def filter_raw(raw, args):
    raw.filter(args.l_freq, args.h_freq)


def preprocessing(raw, data_format, args, task, train=True):
    # Get events
    events, events_id = events_from_annotations(raw)
    if task == 1 or task == 2:
        events_id = dict(right=2, left=3)
    else:
        events_id = dict(hands=2, feet=3)

    picks = pick_types(raw.info, eeg=True)

    # Get epochs and labels (class y) from events
    epochs = Epochs(
            raw, events, events_id,
            args.tmin, args.tmax, picks=picks, proj=True, preload=True)
    y = epochs.events[:, -1] - 2

    if args.plot and train:
        epochs.plot_psd()

    # Get X
    if data_format == 'normal':
        epochs_train = epochs.copy().crop(
                                    tmin=args.crop_min, tmax=args.crop_max)
        X = epochs_train.get_data()
    else:
        psds, freqs = psd_multitaper(
                epochs, low_bias=False,
                tmin=args.crop_min, tmax=args.crop_max,
                picks=picks, proj=True)
        X = psds
    return X, y


def model_training(X, y, args):
    cv = ShuffleSplit(args.iterations, test_size=0.2, random_state=1)
    lda = LinearDiscriminantAnalysis()
    mycsp = myCSP(n_components=args.n_components)
    pipeline = Pipeline([('CSP', mycsp), ('LDA', lda)])
    scores = cross_val_score(pipeline, X, y, cv=cv)
    print("\nScores:", scores)
    print("\nMean classification accuracy: %f" % np.mean(scores))
    save_model(X, y, pipeline, args)


def model_prediction(X, y):
    loaded_model = load_model()

    # Calc of Predictions
    score = loaded_model.score(X, y)
    predict_proba = loaded_model.predict_proba(X)

    # Displaying Results
    predictions = []
    [predictions.append(np.argmax(n)) for n in predict_proba]
    y = y.tolist()
    errors = differences(y, predictions)
    print("\n")
    print(y, "-> Class")
    print(predictions, "-> Predictions")
    print("\nClassification accuracy: %f" % score)
    print("Error(s):", errors)


def rt_prediction_preprocessing(raw, args, task):
    events, events_id = events_from_annotations(raw)
    tmax = raw[-1][-1][-1]
    if task == 1 or task == 2:
        events_id = dict(right=2, left=3)
    else:
        events_id = dict(hands=2, feet=3)
    picks = pick_types(raw.info, eeg=True)

    # Create channel event trigger with annotations
    info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
    stim_data = np.zeros((1, len(raw.times)))
    stim_raw = mne.io.RawArray(stim_data, info)
    raw.add_channels([stim_raw], force_update_info=True)
    raw.add_events(events, stim_channel='STI')

    # Init Mock Real Time
    rt_client = MockRtClient(raw)
    rt_epochs = RtEpochs(
            rt_client, events_id,
            args.tmin, args.tmax, picks=picks, proj=True, stim_channel='STI')
    rt_epochs.start()
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=tmax, buffer_size=1000)

    return rt_epochs


def model_rt_prediction(rt_epochs, data_format, args, filtering):
    loaded_model = load_model()
    predictions = []
    y = rt_epochs.events[:, -1] - 2

    # Calc of Real Time Predictions
    print("\n")
    for ep_num, X in enumerate(rt_epochs.iter_evoked()):
        print("Just got epoch %d" % (ep_num + 1))
        X.crop(args.crop_min, args.crop_max)
        if filtering:
            X.filter(args.l_freq, args.h_freq)
        if data_format == 'normal':
            X = X.data
        else:
            psds, freqs = psd_multitaper(
                    X, low_bias=False, proj=True)
            X = psds
        prediction = np.argmax(loaded_model.predict_proba(X))
        predictions.append(prediction)
        print(prediction)

    # Displaying Results
    y = y.tolist()
    size = len(y)
    errors = differences(predictions, y)
    score = (size - errors) / size
    print("\n")
    print(y, "-> Class")
    print(predictions, "-> Predictions")
    print("\nClassification accuracy: %f" % score)
    print("Error(s):", errors)


def real_time_predictions(raw, args):
    loaded_args = load_args()
    raw.drop_channels(loaded_args.bads)
    data_format = loaded_args.data_format
    pre_filt = args.pre_filter
    task = args.task
    if args.saved_filter:
        args = loaded_args

    # Filtering
    if pre_filt:
        filter_raw(raw, args)

    # Preprocessing
    rt_epochs = rt_prediction_preprocessing(raw, args, task)

    # Real Time Prediction
    model_rt_prediction(rt_epochs, data_format, args, not pre_filt)


def predictions(raw, args):
    loaded_args = load_args()
    raw.drop_channels(loaded_args.bads)
    task = args.task
    data_format = loaded_args.data_format
    if args.saved_filter:
        args = loaded_args

    # Filtering
    filter_raw(raw, args)

    # Preprocessing
    X, y = preprocessing(raw, data_format, args, task, train=False)

    # Prediction
    model_prediction(X, y)


def train(raw, args):
    if args.plot:
        raw.plot(block=True)
        raw.plot_psd()

    # Filtering
    filter_raw(raw, args)

    if args.plot:
        raw.plot(block=True)
        raw.plot_psd()

    args.bads = raw.info['bads']

    # Preprocessing
    X, y = preprocessing(raw, args.data_format, args, args.task)

    # Training
    model_training(X, y, args)


def main():
    mode = {
            'train': train,
            'predict': predictions,
            'rtpredict': real_time_predictions,
    }
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    args = parse()
    raw = get_data(args)
    mode[args.mode](raw, args)


if __name__ == '__main__':
    main()
