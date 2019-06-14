import inspect
import re
import argparse

import numpy as np
import scipy as scp

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.time_frequency import psd_multitaper


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
        eigen_values, eigen_vectors = scp.linalg.eigh(covs[0], covs[0] + covs[1])
        ix = np.argsort(np.abs(eigen_values - 0.5))[::-1] # to select better (n component option)
        eigen_vectors = eigen_vectors[:, ix]
        self.spatialFilters_ = eigen_vectors.T
        self.spatialFilters_ = self.spatialFilters_[:self.n_components]
        return self

    def transform(self, X):
        X_filtred = np.asarray([np.dot(self.spatialFilters_, epoch) for epoch in X])
        X_filtred = np.log(X_filtred.var(axis=2, ddof=1)) # manual var ?
        return X_filtred


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
            help="Select number of runs task",
            type=int,
            choices=range(1, 4),
            default=3)
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
            help="Select the frequencies below which to filter out of the data",
            type=float,
            default=8.)
    parser.add_argument(
            "-hf",
            "--h_freq",
            help="Select the frequencies above which to filter out of the data",
            type=float,
            default=45.)
    parser.add_argument(
            "-ci",
            "--crop_min",
            help="Start time in seconds of the epoch",
            type=float,
            default=0.5)
    parser.add_argument(
            "-ca",
            "--crop_max",
            help="End time in seconds of the epoch",
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
            "-p",
            "--plot",
            help="Plot data",
            action="store_true")
    parser.add_argument(
            '-f',
            '--data_format',
            help='Select data formt',
            choices=['normal', 'psd'],
            default=['normal'])
    args = parser.parse_args()
    return args


def list_runs(args):
    runs = [i for i in np.arange(
        args.task + 2,
        args.task + 2 + (args.n_task) * 4,
        4)]
    return runs


def get_data(args):
    runs = list_runs(args)
    raws = eegbci.load_data(args.subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raws])  # concatenate same tasks
    return raw


def filter_raw(raw, args):
    raw.filter(args.l_freq, args.h_freq)
    if args.plot:
        raw.plot(block=True)
        raw.plot_psd()


def preprocessing(raw, args):
    events, events_id = events_from_annotations(raw)
    if args.task == 1 or args.task == 2:
        events_id = dict(right=2, left=3)
    else:
        events_id = dict(hands=2, feet=3)
    picks = pick_types(raw.info, eeg=True)
    epochs = Epochs(
            raw, events, events_id,
            args.tmin, args.tmax, picks=picks, proj=True, preload=True)
    if args.plot:
        epochs.plot_psd()
    y = epochs.events[:, -1] - 2
    if args.data_format[0] == 'normal':
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


def train(X, y, args):
    cv = ShuffleSplit(args.iterations, test_size=0.2, random_state=1)
    lda = LinearDiscriminantAnalysis()
    mycsp = myCSP(n_components=args.n_components)
    clf = Pipeline([('CSP', mycsp), ('LDA', lda)])
    scores = cross_val_score(clf, X, y, cv=cv)
    print("\n")
    print(scores)
    print("\nMean classification accuracy: %f" % np.mean(scores))


def main():
    args = parse()
    raw = get_data(args)
    if args.plot:
        raw.plot(block=True)
        raw.plot_psd()
    filter_raw(raw, args)
    X, y = preprocessing(raw, args)
    train(X, y, args)


if __name__ == '__main__':
    main()
