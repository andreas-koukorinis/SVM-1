import indicators as ta
import pandas as pd
import pandas.io.data as web
import datetime
import numpy as np
import collections as cs

# Download financial info
def download(source, start, end):
    success = 0
    while success == 0:
        try:
            data = web.DataReader(source, "yahoo", start, end)
            success = 1
        except:
            print "Failed to download " + source + ", retrying..."
    return data

# Search up the data for the given sources and match the dates
def match_dates(sources, start, end):
    ndata = len(sources)
    data = []
    date = []
    intersect = []
    output = []
    for i in xrange(ndata):
        data.append(download(sources[i], start, end))
        date.append(data[i].index)
        if i == 1:
            intersect = list(set(date[1]) & set(date[0]))
        elif i > 1:
            intersect = list(set(date[i]) & set(intersect))
            
    for i in xrange(ndata):
        output.append((data[i].loc[intersect]).sort())
    
    return output

# Return the signals of a particular source
def get_data(data, signals):
    date = data.index
    closep = data["Adj Close"]
    highp =data["High"]
    lowp = data["Low"]
    openp = data["Open"]
    volume = data["Volume"]
    
    collect = []
    inputs, targets = ta.feature_set(date, closep, highp, lowp, openp, volume)
    
    # Cycle through the signals and only return the indicators specified
    for sig in signals:
        for i in xrange(len(inputs)):
            if sig == inputs[i][1]:
                collect.append(inputs[i][0])
                break
    
    features = collect[0]
    for i in range(1, len(collect)):
        features = np.vstack((features, collect[i]))
        
    return features, targets
        
# Set up the labels for the target classes
def set_labels(targets, days):
    labels = np.zeros_like(targets)
    for i in range(days, len(targets)):
        if targets[i] > targets[i-days]:
            labels[i] = 1
        else:
            labels[i] = -1
    return labels[days:]

# Shift the data forward/backward by a given number of days
def shift_data(data, days, type): 
    if type == "forward":
        if len(data.shape) > 1:
            return data[:,days:]
        else:
            return data[days:]
    elif type == "backward":
        if len(data.shape) > 1:
            return data[:,:-days]
        else:
            return data[:-days]

# Stack a number of features
def concatenate(*args):
    ndata = len(args)
    output = args[0]
    for i in range(1, ndata):
        output = np.vstack((output, args[i]))
    return output

def split_data(data, split):
    if len(data.shape) > 1:
        return data[:,:split], data[:,split:]
    else:
        return data[:split], data[split:]
    
def regularise(data):
    output = np.diff(data)
    if len(data.shape) > 1:
        return output / (data[:,:-1] + np.spacing(1))
    else:
        return output / (data[:-1] + np.spacing(1))
            