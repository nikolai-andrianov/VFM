import sys
import csv
from datetime import datetime
import math
import numpy as np
from numpy import array
import pandas as pd
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot, rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, TimeDistributed, Dense, Activation


# Convert the sequence of seconds into datetime array
def parser(x):
    now_seconds = 0
    y = x.astype(np.float) + now_seconds
    z = pd.to_datetime(y, unit='s')
    return z

# Read the contents of the csv file
def read_data(filename):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        headers = next(csvreader)
        vals = list()
        for row in csvreader:
            vals.append(row)

    # Convert vals to numpy array
    data = array(vals).astype(float)

    dt = datetime.today()
    seconds = dt.timestamp()
    dd = datetime.fromtimestamp(seconds)

    data_time = np.zeros(len(data), dtype='datetime64[s]')
    for i in range(len(data)):
        data_time[i] = datetime.fromtimestamp(seconds + data[i, 0])

    return data, headers


# Generate Nts sample input and output sequences from time series in data.
# Each input/output sequence is a 2D array of inp_features/outp_features sampled at inp/outp timesteps.
# Subsequent sequences are indented from each other by step.
# Returns an Nts x inp x inp_features array of input sequences X,
# sampled at timesteps tX, and a 3D array of output sequences Y, sampled at timesteps tY.
def generate_samples_old(data, inp_features, outp_features, Nts, inp, outp, step):

    X = np.zeros((Nts, inp, len(inp_features)))
    tX = np.zeros((Nts, inp))
    Y = np.zeros((Nts, outp, len(outp_features)))
    tY = np.zeros((Nts, outp))
    for i in range(Nts):
        X[i] = data[i*step : i*step+inp, inp_features]
        tX[i] = data[i*step : i*step+inp, 0]
        Y[i] = data[i*step+inp : i*step+inp+outp, outp_features]
        tY[i] = data[i*step+inp : i*step+inp+outp, 0]

    return X, tX, Y, tY

def generate_samples(data, features, Nts, step, length, shift):

    X = np.zeros((Nts, length, len(features)))
    tX = np.zeros((Nts, length))
    for i in range(Nts):
        X[i] = data[i*step+shift : i*step+shift+length, features]
        tX[i] = data[i*step+shift : i*step+shift+length, 0]

    return X, tX

# Visualize data sequences
def visualize_old(data, headers, inp_features, outp_features, Ntr, Nseq, Nts, step, inp, outp):

    Nts = 3     # For simplicity in plotting

    N = data.shape[0]
    Nplots = data.shape[1] - 1  # The first column is time

    # To plot the bars for data sequences
    minf = np.amin(data, axis=0)
    maxf = np.amax(data, axis=0)
    df = maxf - minf
    squeeze = 0.9
    barheight = squeeze * np.minimum(df / Nts, 0.1 * df)
    interbar = 0.1 * barheight
    starty = minf + df/2 + (barheight + interbar) * Nts/2

    f, ax = pyplot.subplots(Nplots, sharex=True)
    for i in range(Nplots):
        ax[i].plot(data[:,0], data[:,i+1])
        ax[i].set_ylabel(headers[i+1])
    #
        # Add bars to indicate the span of data sequences
        startybar = starty[i+1]
        for j in range(Nts, 0, -1):
            endybar = startybar - barheight[i+1]
            ax[i].axhspan(startybar, endybar, xmin=0.25, xmax=0.75, facecolor='0.5', alpha=0.5)   # Input
            startybar = endybar - interbar[i+1]

    ax[0].set_title('Data sequences', fontweight='bold')
    ax[Nplots-1].set_xlabel(headers[0])

# X, tX, Y, tY assumed to be normalized to [0, 1]
def visualize(X, tX, Y, tY):

    Ns = X.shape[0]    # Number of sequences
    Nif = X.shape[2]    # Number of input features
    Nof = Y.shape[2]    # Number of output features

    # To plot the bars for data sequences
    # mint = np.amin(tX, axis=1)
    # maxt = np.amax(tY, axis=1)
    # minfx = np.amin(np.amin(X, axis=1), axis=0)    # Min & max values for input features
    # maxfx = np.amax(np.amax(X, axis=1), axis=0)
    # minfy = np.amin(np.amin(Y, axis=1), axis=0)    # Min & max values for output features
    # maxfy = np.amax(np.amax(Y, axis=1), axis=0)

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
        #
    # Add bars to indicate the span of data sequences
    startybar = starty
    for i in range(Ns):
        endybar = startybar - barheight
        ax.axhspan(startybar, endybar, xmin=min(tX[i,:]), xmax=max(tX[i,:]), facecolor='g', alpha=0.5)  # Input
        ax.axhspan(startybar, endybar, xmin=min(tY[i, :]), xmax=max(tY[i, :]), facecolor='r', alpha=0.5)  # Output
        startybar = endybar - interbar

    ax.set_title('Data sequences', fontweight='bold')
    pyplot.show(block=False)

def read_data(fname, plot_data = False):
    # Read the time series
    # datats = read_csv('riser_pq_200_double.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # datats = read_csv('riser_pq_200.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    #datats = read_csv('riser_pq_uni.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # datats = read_csv('xsin8.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    datats = read_csv(fname, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

    headers = list(datats.columns.values)
    headers.insert(0, datats.index.name)

    # Resample the data using a uniform timestep
    datats = datats.resample('S').mean()
    datats = datats.interpolate(method='linear')

    # Convert data to numpy array
    data = datats.reset_index().values

    # Replace timestamps with seconds
    time_sec = array([data[i, 0].timestamp() for i in range(len(data))])
    data = np.c_[time_sec, data[:, 1:]]

    # Plot the pressure readings
    if (plot_data):
        pyplot.plot(data[:, 0], data[:, 1:8])
        pyplot.xlabel(headers[0])
        pyplot.ylabel('Pressure (bar)')
        # Use the original headers
        # headersplot = [w.replace('x_', '$x_') for w in headers[1:8]]
        # headersplot = [w.replace('}=', '}$=') for w in headersplot]
        # headersplot = [w.replace(' (bar)', '') for w in headersplot]
        # Use the headers p(x=xi)
        headersplot = [w[-8:-2] for w in headers[1:8]]
        px = ['$p(x_{%d}' % i for i in range(1, 8)]
        tail = [')$'] * 7
        headersplot = [px + headersplot + tail for px, headersplot, tail in zip(px, headersplot, tail)]
        pyplot.legend(headersplot)
        pyplot.title('Distributed pressure readings')
        pyplot.show(block=False)
        pyplot.savefig('pressure_readings.pdf')

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data)
    scaler.scale_[0] = 1    # Do not normalize time

    # Apply the same normalization to all pressure readings
    pind = list(range(1, 8))  # Indices of pressure readings
    pmin = scaler.data_min_[pind].min()
    pmax = scaler.data_max_[pind].max()
    scaler.scale_[pind] = ((scaler.feature_range[1] - scaler.feature_range[0]) / (pmax - pmin))
    scaler.min_[pind] = scaler.feature_range[0] - pmin * scaler.scale_[pind]

    data = scaler.transform(data)

    return data, scaler


def run(data, inp_features, outp_features, Nseq, Nts, step, inp, outp, model_name):

    N = data.shape[0]  # Overall sequence length

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

    # Fix the autolayout for matplotlib
    #rcParams.update({'figure.autolayout': True})

    # Model name to save the weights
    ilist = ['%d' % i for i in inp_features]
    ilist = ''.join(ilist)
    olist = ['%d' % i for i in outp_features]
    olist = ''.join(olist)
    #mname = 'model_1core_test_i' + ilist + '_o' + olist
    mname = model_name + '_i' + ilist + '_o' + olist

    model_fit = True    # Fit the model, save it weights to mname.h5, save the convergence history to mname.svg, and run predictions
    #model_fit = False   # Load the previously saved mname.h5 and run predictions

    pyplot.close('all')

    X, tX = generate_samples(data, inp_features, Nts, step, inp, 0)
    Y, tY = generate_samples(data, outp_features, Nts, step, outp, inp)

    visualize(X, tX, Y, tY)

    Nsamples = X.shape[0]

    # Define a 2-layer LSTM model with 20 memory cells in the hidden layers - a bit worse than Ironbell #2
    # Workaround to limit the Dense output just to one feature
    # Y = Y.reshape(len(Y), outp)
    # model = Sequential()
    # model.add(LSTM(20, return_sequences=True, input_shape=( inp, len(inp_features) )))
    # model.add(LSTM(20))
    # model.add(Dense(outp))
    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

    # model = Sequential()
    # model.add(LSTM(20, return_sequences=True, input_shape=( inp, len(inp_features) )))
    # model.add(LSTM(20, return_sequences=True))
    # model.add(TimeDistributed(Dense(1)))

    # Vanilla - the same oscillations...
    # Y = Y.reshape(len(Y), outp)
    # model = Sequential()
    # model.add(LSTM(25, input_shape=( inp, len(inp_features) )))
    # model.add(Dense(outp))
    # model.compile(loss='mae', optimizer='adam')
    # print(model.summary())

    # Ironbell #1 = works for many-to-one, 1 step ahead (i.e. outp=1)
    # Y = Y.reshape(len(Y), outp)
    # model = Sequential()
    # model.add(LSTM(units=10, input_shape=(None, 1), return_sequences=False ))
    # model.add(Dense(1))
    # model.add(keras.layers.Activation('linear'))
    # model.compile(loss='mean_squared_error', optimizer='rmsprop')

    # Ironbell #2 = works for predicting many-to-many when inp=outp (best)
    # model = Sequential()
    # model.add(LSTM(units=300, input_shape=(None, 1), return_sequences=True))
    # model.add(TimeDistributed(Dense(1)))
    # model.add(keras.layers.Activation('linear'))
    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

    # Ironbell #2 modified - as good as the original ironbell #2
    # model = Sequential()
    # model.add(LSTM(units=10, input_shape=(inp, 1), return_sequences=True))
    # model.add(TimeDistributed(Dense(1)))
    # model.add(keras.layers.Activation('linear'))
    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

    if model_name == 'LSTM':

        # Ironbell #2 stacked as per http://danielhnyk.cz
        model = Sequential()
        model.add(LSTM(units=10, input_shape=(inp, len(inp_features)), return_sequences=True))
        model.add(LSTM(units=10, return_sequences=True))
        model.add(LSTM(units=10, return_sequences=True))
        model.add(TimeDistributed(Dense(len(outp_features))))
        model.add(Activation('linear'))
        # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())

        # Ironbell #2 & #3
        history = model.fit(X, Y, batch_size=1, epochs=10, validation_split=0.05)

    elif model_name == 'LSTM_Encoder':

        # Ironbell #3 = Many-to-many with steps_before <> steps_after
        model = Sequential()
        model.add(LSTM(units=10, input_shape=(None, len(inp_features)), return_sequences=False))
        model.add(keras.layers.RepeatVector(outp))
        model.add(LSTM(units=10, return_sequences=True))
        model.add(LSTM(units=10, return_sequences=True))
        model.add(TimeDistributed(Dense(len(outp_features))))
        model.add(keras.layers.Activation('linear'))
        #model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())

        # Ironbell #2 & #3
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

    # Ironbell #3 = Many-to-many with steps_before <> steps_after
    # model = Sequential()
    # model.add(LSTM(units=300, input_shape=(None, len(inp_features)), return_sequences=False))
    # model.add(keras.layers.RepeatVector(outp))
    # model.add(LSTM(units=300, return_sequences=True))
    # model.add(TimeDistributed(Dense(len(outp_features))))
    # model.add(keras.layers.Activation('linear'))
    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])


    # Train the model
    nb_epochs = 200
    nb_batches = Nsamples  # Batch gradient descent
    # history = model.fit(X, Y, batch_size=nb_batches, epochs=nb_epochs, verbose=1)

    # Ironbell #2 & #3
    # history = model.fit(X, Y, batch_size=1, epochs=10, validation_split=0.05)

    # Batch gradient descent - worse than stochatic with batch_size=1
    # history = model.fit(X, Y, batch_size=Nsamples, epochs=nb_epochs, validation_split=0.05)

    # Mini-batch gradient descent - worse than stochatic with batch_size=1
    # history = model.fit(X, Y, batch_size=32, epochs=50, validation_split=0.05)

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

    #pyplot.savefig(mname + '_convergence.svg')
    pyplot.savefig(mname + '_convergence.pdf')


    # Generating the test sequences covering all data, so that the output sequences are not-overlapping
    step = outp
    Ns = int((N-Nseq)/step + 1)
    # Renamed X, tX etc to avoid a bug in PyCharm
    XX, tXX = generate_samples(data, inp_features, Ns, step, inp, 0)
    YY, tYY = generate_samples(data, outp_features, Ns, step, outp, inp)

    if model_name == 'FF':  # Feedforward NN
        XX = XX.reshape(len(XX), inp)

    # Prediction on all data
    Ypred = model.predict(XX, verbose=0)



    pyplot.figure(1)
    for i in range(Ns):
        #pyplot.plot(tXX[i], XX[i], linewidth=3)
        pyplot.plot(tYY[i], YY[i], '+', linewidth=1)
        pyplot.plot(tYY[i], Ypred[i], 'o', linewidth=3)

    #pyplot.legend()
    pyplot.show(block=False)
    #pyplot.show()
    pyplot.savefig(mname + '.svg')
    pyplot.savefig(mname + '.pdf')
    pyplot.savefig(mname + '.png')

    print('Done..')


# -------------------------------------
# Main


# Fix the autolayout for matplotlib
rcParams.update({'figure.autolayout': True})

# Read the data
data, scaler = read_data('riser_pq_uni.csv')    # , plot_data=True

# Prepare training sample sequences from data
N = data.shape[0]  # Overall sequence length
train_frac = 0.5  # Fraction of data used for training
Ntr = int(train_frac * N)  # Estimate the number of timesteps used for training
# Ntr = N# for riser_pq

Nseq = Ntr // 4  # Length of a training sequence

# Ensure and even Nseq to get inp=outp below
if Nseq % 2 != 0:
    Nseq = Nseq - 1

if Nseq < 2 or Nseq > Ntr:
    print('Please set the training sequence length within [2, ' + repr(Ntr) + ']')
    sys.exit(1)

pred_frac = 0.5  # Within a training sequence, fraction of data used for prediction
outp = max(1, int(pred_frac * Nseq))  # Number of timesteps in the output sequence
inp = Nseq - outp  # Number of timesteps in the input training sequence

# Set the number of training sequences => compute the sequence indentation step
# Nts = 5                     # Number of training sequences
# step = int((Ntr - Nseq) / (Nts - 1))  	# Subsequent training sequences are indented from each other by step
# if step > inp:
#     print('Warning: input training sequences are not overlapping!!')

# Set the sequence indentation step => compute the number of training sequences
# step = max(1, inp//2)   # the offset is too big => too few sequences fit until Ntr..
step = 1
Nts = int((Ntr - Nseq) / step + 1)
Ntr = Nseq + step * (Nts - 1)   # Adjust Ntr for the specified step & Nts


# Run forecasts
rf = True
if rf:
    features = range(1, 8)
    for n in features:
        inp_features = [i for i in range(1, n + 1)]
        #inp_features.append(9)
        outp_features = [8, 9]  # Indices of columns in data which will be used as output features
        run(data, inp_features, outp_features, Nseq, Nts, step, inp, outp, 'LSTM')

    # for n in range(1,8):
    #     inp_features = [i for i in range(1,n+1)]       # Indices of columns in data which will be used as input features
    #     inp_features = [1, 2, 3, 4, 5, 9]
    #     outp_features = [9]       # Indices of columns in data which will be used as output features
    #     #run(inp_features, outp_features, 'LSTM')
    #     run(data, inp_features, outp_features, Nseq, Nts, step, inp, outp, 'LSTM')
else:
    # Plot previously saved forecasts
    a = 1



plot_convergence = True
plot_forecasts = False
pyplot.close('all')

pyplot.plot(data[:,0], data[:,9], color='black', label='Ground truth', linewidth=3)

# model_name = 'model_1core'
# model_name = 'FF'
#features = range(1,8)
features = range(7,0,-1)    # For better visualization
#features = range(8,-1,-1)    # 7 <-> FF, -1 <-> LSTM with q & p
#features = [1]
nfeatures = len(features)
#nfeatures = len(features) + 1   # To account for ql
nl = 0
for n in features:

    # if n == 8:
    #     inp_features = [1]
    #     outp_features = [9]
    #     model_name = 'FF'
    # elif n == 0:
    #     inp_features = [1, 2, 3, 4, 5, 9]
    #     outp_features = [9]
    #     model_name = 'LSTM'
    # else:
    #     inp_features = [i for i in range(1,n+1)]       # Indices of columns in data which will be used as input features
    #     outp_features = [9]       # Indices of columns in data which will be used as output features
    #     model_name = 'model_1core'

    inp_features = [i for i in range(1, n + 1)]  # Indices of columns in data which will be used as input features
    outp_features = [8, 9]       # Indices of columns in data which will be used as output features
    model_name = 'LSTM'
    #model_name = 'LSTM_Encoder'
    #model_name = 'model_1core'

    # Model name to plot
    ilist = ['%d' % i for i in inp_features]
    ilist = ''.join(ilist)
    olist = ['%d' % i for i in outp_features]
    olist = ''.join(olist)
    mname = model_name + '_i' + ilist + '_o' + olist
    #mname = model_name + '_i' + ilist + '9_o' + olist

    # Plot label
    # plabel = ['p($x_%d $), ' % i for i in inp_features]
    # plabel= ''.join(plabel)
    # plabel = plabel[:-2]
    # plabel = 'Forecast using ' + plabel

    if len(inp_features) == 1:
        plabel = '{' + str(inp_features[0]) + '}'
    elif len(inp_features) == 2:
        plabel = '{' + str(inp_features[0]) + ', ' + str(inp_features[-1]) + '}'
    else:
        plabel = '{' + str(inp_features[0]) + ', \\dots, ' + str(inp_features[-1]) + '}'

    # if n == 8:
    #     plabel = 'FF using $p(x_' + plabel + ')$'
    # elif n == 0:
    #     plabel = 'LSTM using $p(x_' + plabel + ')$ & $q_l$'
    # else:
    #     plabel = 'LSTM using $p(x_' + plabel + ')'

    plabel = 'LSTM using $p(x_' + plabel + ')$'
    #plabel = 'LSTM Encoder-Decoder using $p(x_' + plabel + ')$'
    #plabel = 'LSTM using $p(x_' + plabel + ')$ & $q_l$'

    # Read the convergence history from *.csv files and keep it in convhist
    if plot_convergence:
        cvh = read_csv(mname + '_convergence.csv', header=0, index_col=0, squeeze=True)
        cvh = cvh.reset_index().values
        if nl == 0:
            convhist = cvh
            convhist_label = array([plabel])
        else:
            convhist = np.c_[convhist, cvh[:, 1]]
            convhist_label = np.append(convhist_label, plabel)

    if plot_forecasts:
        aa = 1
    if 0:
        # Load the previously saved Keras model
        model = load_model(mname + '.h5')

        # Generating the test sequences covering all data
        step = outp        # Non-overlapping output sequences
        #step = outp // 2    # Overlapping output sequences
        Ns = int((N-Nseq)/step + 1)
        seqlabel = ['Sequence %d' % i for i in range(Ns)]
        # Renamed X, tX etc to avoid a bug in PyCharm
        XX, tXX = generate_samples(data, inp_features, Ns, step, inp, 0)
        YY, tYY = generate_samples(data, outp_features, Ns, step, outp, inp)

        if model_name == 'FF':  # Feedforward NN
            XX = XX.reshape(len(XX), inp)

        # Prediction on all data
        Ypred = model.predict(XX, verbose=0)

        # Plot predictions
        pyplot.figure(1)
        for i in range(Ns):

            # Direct coloring all readings
            # if i == 0:
            #     pyplot.plot(tYY[i], Ypred[i], color=(( nl/(nfeatures-1), 0, 1 - nl/(nfeatures-1) )), label=plabel)
            # else:
            #     pyplot.plot(tYY[i], Ypred[i], color=(( nl/(nfeatures-1), 0, 1 - nl/(nfeatures-1) )))

            # Inverse coloring all readings
            if i == 0:
                pyplot.plot(tYY[i], Ypred[i], color=(( 1 - nl/(nfeatures-1), 0, nl/(nfeatures-1) )), label=plabel)
            else:
                pyplot.plot(tYY[i], Ypred[i], color=(( 1 - nl/(nfeatures-1), 0,   nl/(nfeatures-1) )))

            #  Single reading with sequences legends & colors
            #pyplot.plot(tYY[i], Ypred[i], label=seqlabel[i])

            #  Single reading with single legend & color
            # if i == 0:
            #     pyplot.plot(tYY[i], Ypred[i], color=((0,0,1)), label=plabel)
            # else:
            #     pyplot.plot(tYY[i], Ypred[i], color=((0, 0, 1)))

    else:
        b = 1

    nl += 1

pyplot.xlabel('Time (sec)')
pyplot.ylabel('Normalized liquid rate')
pyplot.title('Liquid rate forecast')
pyplot.legend()
pyplot.show(block=False)
pyplot.savefig('forecast_pressure_reverse.pdf')

# Plot convergence history
if plot_convergence:
    pyplot.figure(2)
    for i in range(nfeatures):
        pyplot.semilogy(convhist[:, 0], convhist[:, i+1], color=(( 1 - i/(nfeatures-1), 0, i/(nfeatures-1) )), label=convhist_label[i])

    pyplot.xlabel('Epoch')
    pyplot.ylabel('MSE')
    pyplot.ylim(0.001, 0.02)    # To compare convergence histories
    pyplot.title('Convergence history')
    pyplot.legend()
    pyplot.show(block=False)
    pyplot.savefig('conv_history.pdf')



