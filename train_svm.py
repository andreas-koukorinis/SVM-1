import process_data as data_proc
import indicators as ta
import pandas.io.data as web
import datetime
import itertools as it
import numpy as np
from sklearn import cross_validation as cv
from sklearn import svm

def get_score(predicted, targets):
    true = 0.0
    false = 0.0
    ndata= len(predicted)
    for i in xrange(ndata):
        if predicted[i] == targets[i]:
            true += 1
        else:
            false += 1
    return true / (ndata + false)

def grid_search(train_labels, train_feats, type):
    bestcv = -np.inf
    bestc = 0
    bestg = 0
    accuracy = 0
    for log2c in range(-8,8):
        for log2g in range(-8,8):
            c = np.power(2.0, log2c)
            g = np.power(2.0, log2g)
            if type == "regression":
                model = svm.SVR(kernel='rbf', C = c, gamma = g)
                scores = cv.cross_val_score(model, train_feats, train_labels, cv=5)
                accuracy = np.mean(scores)
            else:
                model = svm.SVC(kernel='rbf', C = c, gamma = g)
                scores = cv.cross_val_score(model, train_feats, train_labels, cv=5,scoring="accuracy")
                accuracy = np.mean(scores)
            if accuracy >= bestcv:
                bestcv = accuracy 
                bestc = c 
                bestg = g
    
    return bestcv, bestc, bestg

def train_svm(train_feats, train_labels, test_feats, test_labels, type):
            bestcv, bestc, bestg = grid_search(train_labels, train_feats, type)
            
            model = []
            if type == "regression":
                model = svm.SVR(kernel='rbf', C = bestc, gamma = bestg)
            else:
                model = svm.SVC(kernel='rbf', C = bestc, gamma = bestg)
            model.fit(train_feats, train_labels)
            predicted = model.predict(test_feats)
            score = get_score(predicted, test_labels)
            
            return score, bestcv, bestc, bestg, predicted

#--- Main function ---#
if __name__ == "__main__":
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2014, 1, 1)
    
    # Set up the list of sources and their desired indicators
    sources = ["^GSPC","^GDAXI","^FTSE","^N225","Oil"]
    sig_sp = ["EMA7","EMA50","EMA200","RSI","ADX","MACD","Signal","Derivative","High","Low","Close","Volume"]
    sig_other = ["Close"]
    
    # Get the data for all information sources
    data = data_proc.match_dates(sources, start, end)
    
    # First get the S&P inputs and targets and then the others:
    input_sp, targets = data_proc.get_data(data[0], sig_sp)
    input_dax, _ = data_proc.get_data(data[1], sig_other)
    input_ftse, _ = data_proc.get_data(data[2], sig_other)
    input_n225, _ = data_proc.get_data(data[3], sig_other)
    input_oil, _ = data_proc.get_data(data[4], sig_other)
    
    # Inputs must be altered slightly so that the nikkei index matches with the others
    input_sp = data_proc.shift_data(input_sp,1,"backward")
    input_dax = data_proc.shift_data(input_dax,1,"backward")
    input_ftse = data_proc.shift_data(input_ftse,1,"backward")
    input_n225 = data_proc.shift_data(input_n225,1,"forward")
    input_oil = data_proc.shift_data(input_oil,1,"backward")
    
    
    # Put all inputs together
    features = data_proc.concatenate(input_sp, input_dax, input_ftse, input_n225, input_oil)
    ndata = features.shape[1]
    nfeats = features.shape[0]
    
    # How many data points to train / test with?
    split = ndata - 50
    
    best_day = 0
    best_score = -np.inf
    best_c = 0
    best_g = 0
    best_features = []
    best_pred = []
    target_labels = []
    
    # Test all combinations of five features
    feat_list = xrange(nfeats)
    type = "classify"
    best_stats = open("/Users/admin/Documents/IDE/Eclipse/SVM/best_stats.txt", "w")
    
    for count in range(1, 6):
        print "Testing feature combinations of size " + str(count)
        grp_count = 0
        size = len(list(it.combinations(feat_list, count)))
        for f in it.combinations(feat_list, count):
            grp_count += 1
            print "Testing group " + str(grp_count) + " of " + str(size)
            inputs = features[f[0], :]
        
            for i in range(1, len(f)):
                inputs = np.vstack((inputs, features[f[i],:]))
        
            for days in range(1, 5):
                # First set up the correct range
                if type == "classify":
                    labels = data_proc.set_labels(targets, days)
                else:
                    labels = targets[days:]
                    
                if days > 1:
                    if len(inputs.shape) > 1:
                        inputs = inputs[:,:-days+1]
                    else:
                        inputs = inputs[:-days+1]
                
                train_feats, test_feats = data_proc.split_data(inputs, split)
                train_labels, test_labels = data_proc.split_data(labels, split)
                
                score, bestcv, bestc, bestg, predicted = train_svm(np.asmatrix(train_feats).T, train_labels, 
                                                                   np.asmatrix(test_feats).T, test_labels, type)
                
                if score > best_score:
                    best_pred = predicted 
                    best_score = score
                    best_day = days
                    best_c = bestc
                    best_g = bestg
                    best_features = f
                    target_labels = test_labels
                    best_stats.write("Current best accuracy: " + str(best_score) + "\n")
                    best_stats.write("Features: " + str(best_features) + "\n")
                    best_stats.write("Cost: " + str(best_c) + "\n")
                    best_stats.write("Gamma: " + str(best_g) + "\n")
                    best_stats.write("Lookahead: " + str(best_day) + "\n")
                    best_stats.write("\n\n")
                    best_stats.flush()
     
        
    print "Best feature: " + str(best_features)
    print "Best score: " + str(best_score)
    print "Best lookahead: " + str(best_day)
    print "Cost: " + str(best_c)
    print "Gamma: " + str(best_g)
    
    np.savetxt("/Users/admin/Documents/IDE/Eclipse/SVM/output.txt", best_pred)
    np.savetxt("/Users/admin/Documents/IDE/Eclipse/SVM/targets.txt", target_labels)
    best_stats.close()
        
        
        
        
        
#     ex = "DAX"
#     suffix = ".DE"
#     root = "/Users/admin/Documents/MATLAB/Data/" + ex
#     start = datetime.datetime(2010, 1, 1)
#     end = datetime.datetime(2014, 1, 1)
#     
#     with open(root + "/symbols.txt") as temp_file:
#         symbols = [line.rstrip('\n') for line in temp_file]
#         
#     for sym in symbols:
#         success = 0
#         while success == 0:
#             try:
#                 data = web.DataReader(sym + suffix, "yahoo", start, end)
#                 success = 1
#             except:
#                 print "Failed to gather data, retrying..."
#         
#         date = data.index
#         closep = data["Adj Close"]
#         highp = data["High"]
#         lowp = data["Low"]
#         openp = data["Open"]
#         volume = data["Volume"]
#         
#         inputs, outputs = ta.feature_set(date, closep, highp, lowp, openp, volume)
#         labels = np.asmatrix(np.zeros_like(outputs))
#         
#         if np.sum(np.isinf(inputs)) > 0 or np.sum(np.isnan(inputs)) > 0:
#             print "Skipping " + sym
#             continue
#             
