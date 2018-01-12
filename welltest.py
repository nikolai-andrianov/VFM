import sys
#import csv
import datetime as dttm
import math
import numpy as np
from numpy import array
import pandas as pd
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot, rcParams, dates
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, TimeDistributed, Dense, Activation


def read_data(fname, plot_data = False):
    # Read the time series
    datats = read_csv(fname, header=0, dayfirst=True, parse_dates=[0], index_col=0, squeeze=True)  # , date_parser=parser

    headers = list(datats.columns.values)
    headers.insert(0, datats.index.name)

    # Convert data to numpy array
    data = datats.reset_index().values

    # Split data into flow periods, and resample each flow period using a uniform timestep
    dt = np.ediff1d(data[:, 0])
    fpbreak = dttm.timedelta(hours=1)  # Minimal break between flow periods
    dt = dt - fpbreak
    ind = np.where(dt - fpbreak > pd.Timedelta(0))[0]
    ind = np.r_[ind, len(data)-1]

    Nfp = len(ind)  # Number of flow periods
    fp = ['None'] * Nfp
    n0 = 0
    n1 = ind[0]+1
    for n in range(Nfp):
        # Resample each flow period separately
        fpts = datats[n0:n1].resample('T').mean()
        fpts = fpts.interpolate(method='linear')
        # Save the resampled flow period to a list of numpy arrays
        fp[n] = fpts.reset_index().values
        #fp[n] = data[n0:n1,:]
        n0 = n1
        if n+1 < Nfp:
            n1 = ind[n+1] + 1

    # Plot the graphs
    if (plot_data):

        color = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
        dfmt = dates.DateFormatter('%b %d') # Month day

        # Pressure and temperature
        fig, ax1 = pyplot.subplots()
        ax2 = ax1.twinx()
        for n in range(Nfp):
            if n == 0:
                hl1 = ax1.plot(fp[n][:, 0], fp[n][:, 1], color=color[3], label='Pressure')
                hl2 = ax2.plot(fp[n][:, 0], fp[n][:, 2], color=color[4], label='Temperature')
            else:
                ax1.plot(fp[n][:, 0], fp[n][:, 1], color=color[3])
                ax2.plot(fp[n][:, 0], fp[n][:, 2], color=color[4])

        ax1.xaxis.set_major_formatter(dfmt)
        fig.autofmt_xdate()
        ax1.set_ylabel(headers[1], color=color[3])
        ax1.tick_params(axis='y', colors=color[3])
        headers[2] = headers[2].replace('degC', '°C')
        ax2.set_ylabel(headers[2], color=color[4])
        ax2.tick_params(axis='y', colors=color[4])

        hl = hl1 + hl2
        labs = [h.get_label() for h in hl]
        ax1.legend(hl, labs, loc=2)
        pyplot.title('Pressure and temperature data')
        pyplot.show(block=False)
        pyplot.savefig('wt_PT.pdf')

        # Flow rates
        fig, ax1 = pyplot.subplots()
        ax2 = ax1.twinx()
        for n in range(Nfp):
            if n == 0:
                hl1 = ax1.plot(fp[n][:, 0], fp[n][:, 3], color=color[1], label='Oil rate')
                hl2 = ax1.plot(fp[n][:, 0], fp[n][:, 4], color=color[0], label='Water rate')
                hl3 = ax2.plot(fp[n][:, 0], fp[n][:, 5], color=color[2], label='Gas rate')
            else:
                ax1.plot(fp[n][:, 0], fp[n][:, 3], color=color[1])
                ax1.plot(fp[n][:, 0], fp[n][:, 4], color=color[0])
                ax2.plot(fp[n][:, 0], fp[n][:, 5], color=color[2])

        ax1.xaxis.set_major_formatter(dfmt)
        fig.autofmt_xdate()
        rheader = headers[3].split()[0] + ' & ' + headers[4]
        ax1.set_ylabel(rheader, color=color[1])
        ax1.tick_params(axis='y', colors=color[1])
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.set_ylabel(headers[5], color=color[2])
        ax2.tick_params(axis='y', colors=color[2])

        hl = hl1 + hl2 + hl3
        labs = [h.get_label() for h in hl]
        ax1.legend(hl, labs, loc=1)
        pyplot.title('Flow rates data')
        pyplot.show(block=False)
        pyplot.savefig('wt_Q.pdf')

    # Get the normalization parameters for all data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data[:,1:]) # Exclude Datetime from normalization

    # Normalize every flow period
    for n in range(Nfp):
        fp[n][:,1:] = scaler.transform(fp[n][:,1:])

    return fp, headers, scaler

# Define sequences, shifted by step, for all flow periods fp
def define_fp_seq(fp, step, verbose=False):

    Nseqmin = max([fp[n].shape[0] for n in FP])
    for n in FP:
        N = fp[n].shape[0]  # Sequence length for the n-th training flow period
        train_frac = 1  # Fraction of data used for training
        Ntr = int(train_frac * N)  # Estimate the number of timesteps used for training

        Nseq = Ntr // 4  # Length of a training sequence

        # Ensure even Nseq to get inp=outp below
        if Nseq % 2 != 0:
            Nseq = Nseq - 1

        if Nseq < 2 or Nseq > Ntr:
            print('Please set the training sequence length within [2, ' + repr(Ntr) + ']')
            sys.exit(1)

        pred_frac = 0.5  # Within a training sequence, fraction of data used for prediction
        outp = max(1, int(pred_frac * Nseq))  # Number of timesteps in the output sequence
        inp = Nseq - outp  # Number of timesteps in the input training sequence

        # Compute the number of training sequences
        Nts = int((Ntr - Nseq) / step + 1)
        Ntr = Nseq + step * (Nts - 1)  # Adjust Ntr for the specified step & Nts

        if verbose:
            print('Flow period ' + str(n) + ':')
            print('     Length of a training sequence: ' + str(Nseq))
            print('     Number of training sequences: ' + str(Nts))
            print('     Sequence indentation step: ' + str(step))

        if Nseq < Nseqmin:
            Nseqmin = Nseq
            nmin = n

    # Choose the minimal Nseq
    Nseq = Nseqmin
    if verbose:
        print('Choosing min training sequence length of ' + str(Nseq) + ' from flow period ' + str(nmin))

    pred_frac = 0.5  # Within a training sequence, fraction of data used for prediction
    outp = max(1, int(pred_frac * Nseq))  # Number of timesteps in the output sequence
    inp = Nseq - outp  # Number of timesteps in the input training sequence

    # Create a list of Nts for all flow periods
    Ntsfp = np.zeros(Nfp, dtype=np.int)
    for n in FP:
        N = fp[n].shape[0]  # Sequence length for the n-th flow period
        train_frac = 1  # Fraction of data used for training
        Ntr = int(train_frac * N)  # Estimate the number of timesteps used for training
        Ntsfp[n] = int((Ntr - Nseq) / step + 1)

    return Nseq, Ntsfp, inp, outp


def generate_samples(data, features, Nts, step, length, shift):

    X = np.zeros((Nts, length, len(features)))
    tX = np.tile(data[0,0], (Nts, length))     # Create a 2D timestamp array

    for i in range(Nts):
        X[i] = data[i*step+shift : i*step+shift+length, features]
        tX[i] = data[i*step+shift : i*step+shift+length, 0]

    return X, tX

# X, tX, Y, tY assumed to be normalized to [0, 1]
def visualize(X, tX, Y, tY):

    Ns = X.shape[0]    # Number of sequences
    Nif = X.shape[2]    # Number of input features
    Nof = Y.shape[2]    # Number of output features

    pyplot.close('all')

    # Plot input sequences
    squeeze = 0.9
    barheight = squeeze * np.minimum(1 / Ns, 0.1)
    interbar = 0.1 * barheight
    starty = 0.5 + (barheight + interbar) * Ns / 2

    f, ax = pyplot.subplots(1, sharex=True)
    pyplot.xlim(0, 1)   # Fix the x range to (0, 1)

    for i in range(Ns):
        for j in range(Nif):
            ax.plot(tX[i,:], X[i,:,j], 'b')
        for j in range(Nof):
            ax.plot(tY[i, :], Y[i, :, j], 'r')

    # Add bars to indicate the span of data sequences
    startybar = starty
    for i in range(Ns):
        endybar = startybar - barheight
        ax.axhspan(startybar, endybar, xmin=min(tX[i,:]), xmax=max(tX[i,:]), facecolor='g', alpha=0.5)  # Input
        ax.axhspan(startybar, endybar, xmin=min(tY[i, :]), xmax=max(tY[i, :]), facecolor='r', alpha=0.5)  # Output
        startybar = endybar - interbar

    ax.set_title('Data sequences', fontweight='bold')
    pyplot.show(block=False)

def run(fp, TFP, inp_features, outp_features, Nseq, Ntsfp, step, inp, outp, model_name):

    # Fix random seed for reproducibility
    import os
    import random
    import tensorflow
    import keras
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1)
    random.seed(1)
    session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    tensorflow.set_random_seed(1)
    sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)

    # Model name to save the weights
    ilist = ['%d' % i for i in inp_features]
    ilist = ''.join(ilist)
    olist = ['%d' % i for i in outp_features]
    olist = ''.join(olist)
    mname = model_name + '_i' + ilist + '_o' + olist

    # Generate training sequences for selected flow periods
    for n in TFP:
        _X, _tX = generate_samples(fp[n], inp_features, Ntsfp[n], step, inp, 0)
        _Y, _tY = generate_samples(fp[n], outp_features, Ntsfp[n], step, outp, inp)
        # Accumulate sequences from all training periods
        if n == 0:
            X, tX = _X, _tX
            Y, tY = _Y, _tY
        else:
            X = np.append(X, _X, axis=0)
            tX = np.append(tX, _tX, axis=0)
            Y = np.append(Y, _Y, axis=0)
            tY = np.append(tY, _tY, axis=0)


    if model_name == 'LSTM':

        model = Sequential()
        model.add(LSTM(units=10, input_shape=(inp, len(inp_features)), return_sequences=True))
        model.add(LSTM(units=10, return_sequences=True))
        model.add(LSTM(units=10, return_sequences=True))
        model.add(TimeDistributed(Dense(len(outp_features))))
        model.add(Activation('linear'))
        # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
        model.compile(loss='mean_squared_error', optimizer='adam')

        history = model.fit(X, Y, batch_size=1, epochs=10, validation_split=0.05)

    elif model_name == 'FF':  # Feedforward NN

        if len(inp_features) != 1 or len(outp_features) != 1:
            print('Feedforward NN is only defined for a single feature.. Exiting..')
            return

        X = X.reshape(len(X), inp)
        Y = Y.reshape(len(Y), outp)

        model = Sequential()
        model.add(Dense(10, input_shape=(inp,)))
        model.add(Dense(10))
        model.add(Dense(10))
        model.add(Dense(outp))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())

        history = model.fit(X, Y, batch_size=1, epochs=10, validation_split=0.05)

    else:
        print('Model not defined.. Exiting..')
        return


    # Save the model
    model.save(mname + '.h5')

    # Plotting the convergence history
    pyplot.figure(3)
    pyplot.semilogy(history.history['loss'])
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.show(block=False)

    # Save the convergence history
    df = pd.DataFrame(history.history['loss'])
    df.index.name = 'Epoch'
    df.to_csv(mname + '_convergence.csv', header=['loss'])
    pyplot.savefig(mname + '_convergence.pdf')

    print('Done..')

    return model


# -------------------------------------
# Main


# Fix the autolayout for matplotlib
rcParams.update({'figure.autolayout': True})

# Read and normalize the flow periods
fp, headers, scaler = read_data('welltest.csv')    # , plot_data=True

inp_features = [1, 2]       # Indices of columns in data which will be used as output features
outp_features = [3, 4, 5]   # Indices of columns in data which will be used as output features

TFP = [0, 1]    # Indices of flow periods, used for training
Ntfp = len(TFP) # Number of flow periods, used for training

Nfp = len(fp)   # Number of flow periods
FP = list(range(Nfp))

errFP = set(TFP) - set(FP)
if Ntfp > Nfp:
    print('Too many training flow periods.. Exiting..')
    sys.exit(1)
if len(errFP) > 0:
    print('Incorrect training flow period(s): ' + errFP + ' Exiting..')
    sys.exit(1)

# Define sequences, shifted by step, for all flow periods fp
step = 1
Nseq, Ntsfp, inp, outp = define_fp_seq(fp, step)

# Compute the relative forecasting interval
DT = pd.Timedelta(0)
for i in TFP:
    t0 = fp[i][0,0]
    t1 = fp[i][-1, 0]
    dt = t1 - t0
    DT += dt

DT = DT.days * 24 * 60 + DT.seconds / 60    # Convert DT to minutes
print('Relative forecasting interval: ' + str(outp/DT*100) + '%')

# File name for the model
model_name = 'LSTM'
ilist = ['%d' % i for i in inp_features]
ilist = ''.join(ilist)
olist = ['%d' % i for i in outp_features]
olist = ''.join(olist)
tfplist = ['%d' % i for i in TFP]
tfplist = ''.join(tfplist)
#mname = model_name + '_i' + ilist + '_o' + olist + '_FP' + str(tfplist)
mname = model_name + '_i' + ilist + '_o' + olist

# Get the trained Keras model
train_model = True
if train_model:
    model = run(fp, TFP, inp_features, outp_features, Nseq, Ntsfp, step, inp, outp, model_name)
else:   # Load the previously saved Keras model
    model = load_model(mname + '.h5')

# Running predictions with the model
#step = outp     # Adjusting number of non-overlapping sequences in flow periods
step = outp // 2     # Overlap sequences in flow periods
Nsfp = np.zeros(Nfp, dtype=np.int)
for n in FP:
    N = fp[n].shape[0]  # Sequence length for the n-th flow period
    Nsfp[n] = int((N - Nseq) / step + 1)    # Number of sequences of length Nseq in fp[n]

# Generating the test sequences covering all data, so that the output sequences are non-overlapping
for n in FP:
    _X, _tX = generate_samples(fp[n], inp_features, Nsfp[n], step, inp, 0)
    _Y, _tY = generate_samples(fp[n], outp_features, Nsfp[n], step, outp, inp)
    _Yplot, _tplot = generate_samples(fp[n], outp_features, Nsfp[n], step, inp, 0)  # To fill the gap in plotting forecasts at the beginning of each flow period
    # Accumulate sequences from all training periods
    if n == 0:
        X, tX = _X, _tX
        Y, tY = _Y, _tY
        Yplot, tplot = _Yplot, _tplot
    else:
        X = np.append(X, _X, axis=0)
        tX = np.append(tX, _tX, axis=0)
        Y = np.append(Y, _Y, axis=0)
        tY = np.append(tY, _tY, axis=0)
        Yplot = np.append(Yplot, _Yplot, axis=0)
        tplot = np.append(tplot, _tplot, axis=0)

# Prediction on all data
Ypred = model.predict(X, verbose=0)

# Get back the dimensional rates
Y -= scaler.min_[2:]
Y /= scaler.scale_[2:]
Ypred -= scaler.min_[2:]
Ypred /= scaler.scale_[2:]
Yplot -= scaler.min_[2:]
Yplot /= scaler.scale_[2:]

# Plotting parameters
color = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
dfmt = dates.DateFormatter('%b %d')  # Month day
fig, ax1 = pyplot.subplots()
ax2 = ax1.twinx()

Ns = X.shape[0]
opacity = 0.2
for i in range(Ns):

    # Filling the gap in plotting forecasts at the beginning of each flow period
    ax1.plot(tplot[i], Yplot[i, :, 0], color=color[1], zorder=0)
    ax1.plot(tplot[i], Yplot[i, :, 1], color=color[0], zorder=0)
    ax2.plot(tplot[i], Yplot[i, :, 2], color=color[8], zorder=0)

    if i == 0:
        hl1 = ax1.plot(tY[i], Y[i, :, 0], color=color[1], zorder=0, label='Measured Qo')
        hl2 = ax1.plot(tY[i], Y[i, :, 1], color=color[0], zorder=0, label='Measured Qw')
        hl3 = ax2.plot(tY[i], Y[i, :, 2], color=color[8], zorder=0, label='Measured Qg')

        hl4 = ax1.plot(tY[i, step:], Ypred[i, step:, 0], 'r', zorder=1, linewidth=3, label='Forecasted Qo')
        hl5 = ax1.plot(tY[i, step:], Ypred[i, step:, 1], 'b', zorder=1, linewidth=3, label='Forecasted Qw')
        hl6 = ax2.plot(tY[i, step:], Ypred[i, step:, 2], 'g', zorder=1, linewidth=3, label='Forecasted Qg')

        ax1.plot(tY[i, :step], Ypred[i, :step, 0], 'r', zorder=1, alpha=opacity, linewidth=3)   # Semi-transparent oil rate
        ax1.plot(tY[i, :step], Ypred[i, :step, 1], 'b', zorder=1, alpha=opacity, linewidth=3)   # Semi-transparent water rate
        ax2.plot(tY[i, :step], Ypred[i, :step, 2], 'g', zorder=1, alpha=opacity, linewidth=3)   # Semi-transparent gas rate

    else:
        ax1.plot(tY[i], Y[i, :, 0], color=color[1], zorder=0)  # 'Oil rate'
        ax1.plot(tY[i], Y[i, :, 1], color=color[0], zorder=0)  # 'Water rate'
        ax2.plot(tY[i], Y[i, :, 2], color=color[8], zorder=0)  # 'Gas rate'

        ax1.plot(tY[i, :step], Ypred[i, :step, 0], 'r', zorder=1, alpha=opacity, linewidth=3)  # Semi-transparent oil rate
        ax1.plot(tY[i, :step], Ypred[i, :step, 1], 'b', zorder=1, alpha=opacity, linewidth=3)  # Semi-transparent water rate
        ax2.plot(tY[i, :step], Ypred[i, :step, 2], 'g', zorder=1, alpha=opacity, linewidth=3)  # Semi-transparent gas rate

        ax1.plot(tY[i, step:], Ypred[i, step:, 0], 'r', zorder=1, linewidth=3)  # 'Oil rate'
        ax1.plot(tY[i, step:], Ypred[i, step:, 1], 'b', zorder=1, linewidth=3)  # 'Water rate'
        ax2.plot(tY[i, step:], Ypred[i, step:, 2], 'g', zorder=1, linewidth=3)  # 'Gas rate'


ax1.xaxis.set_major_formatter(dfmt)
fig.autofmt_xdate()
rheader = headers[3].split()[0] + ' & ' + headers[4]
ax1.set_ylabel(rheader, color=color[1])
ax1.tick_params(axis='y', colors=color[1])
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax2.set_ylabel(headers[5], color=color[2])
ax2.tick_params(axis='y', colors=color[2])

hl = hl1 + hl2 + hl3 + hl4 + hl5 + hl6
labs = [h.get_label() for h in hl]
ax1.legend(hl, labs, loc=1)
pyplot.title('Flow rates data')
pyplot.show(block=False)
pyplot.savefig('wt_Q_forecast.pdf')

