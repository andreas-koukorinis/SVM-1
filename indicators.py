import numpy as np

# Exponential moving average
def EMA(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    
    a =  np.convolve(values, weights)[:len(values)]
    a[:window] = a[window]
    return a

# Moving average convergence/divergence
def MACD(data, fast=12, slow=26):
    macd = EMA(data, fast) - EMA(data, slow)
    signal = EMA(macd, 9)
    divergence = macd - signal
    
    return macd, signal, divergence

# Relative strength index
def RSI(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down;
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)
    
    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up*(n-1) + upval) / n
        down = (down*(n-1) + downval) / n
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    
    return rsi

# Average directional index helper functions
def TR(d,c,h,l,o,yc):
    x = h-l
    y = abs(h-yc)
    z = abs(l-yc)
    
    if y <= x >= z:
        TR = x
    elif x <= y >= z:
        TR = y
    elif x <= z >= y:
        TR = z
    
    return d, TR


def DM(d,o,h,l,c,yo,yh,yl,yc):
    moveUp = h-yh
    moveDown = yl-l

    if 0 < moveUp > moveDown:
        PDM = moveUp
    else:
        PDM = 0
    if 0 < moveDown > moveUp:
        NDM = moveDown
    else:
        NDM = 0

    return d,PDM,NDM


def calcDIs(date,closep,highp,lowp,openp,volume):
    x = 0
    TRDates = []
    TrueRanges = []
    PosDMs = []
    NegDMs = []

    while x < len(date):
        TRDate, TrueRange = TR(date[x],closep[x],highp[x],lowp[x],openp[x],closep[x-1])
        TRDates.append(TRDate)
        TrueRanges.append(TrueRange)
        
        _,PosDM,NegDM = DM(date[x],openp[x],highp[x],lowp[x],closep[x],openp[x-1],highp[x-1],lowp[x-1],closep[x-1])
        PosDMs.append(PosDM)
        NegDMs.append(NegDM)

        x+=1

    expPosDM = EMA(PosDMs,14)
    expNegDM = EMA(NegDMs,14)
    ATR = EMA(TrueRanges,14)

    ix = 0
    PDIs = []
    NDIs = []

    while ix < len(ATR):
        PDI = 100*(expPosDM[ix]/ATR[ix])
        PDIs.append(PDI)

        NDI = 100*(expNegDM[ix]/ATR[ix])
        NDIs.append(NDI)
        
        ix += 1

    return PDIs,NDIs

# Average directional index
def ADX(date,closep,highp,lowp,openp,volume):
    PositiveDI,NegativeDI = calcDIs(date,closep,highp,lowp,openp,volume)
    ix = 0
    DXs = []

    while ix < len(date):
        DX = 100*( (abs(PositiveDI[ix]-NegativeDI[ix])
                    /(PositiveDI[ix]+NegativeDI[ix])))
        DXs.append(DX)
        ix += 1

    ADX = EMA(DXs,14)
    
    return ADX

def concatenate(*args):
    ndata = len(args)
    output = args[0]
    for i in range(1, ndata):
        output = np.vstack((output, args[i]))
    return output

def set_var(data):
    #output = np.asmatrix(data)
    return data / np.max(abs(data))

# Return the feature set as a list of objects
def feature_set(date,closep,highp,lowp,openp,volume):
    ema7 = set_var(EMA(closep, 7))
    ema50 = set_var(EMA(closep, 50))
    ema200 = set_var(EMA(closep, 200))
    rsi = set_var(RSI(closep, n=14))
    adx = set_var(ADX(date,closep,highp,lowp,openp,volume))
    macd,signal,der = MACD(closep, fast=12, slow=26)
    macd = set_var(macd)
    signal = set_var(signal)
    der = set_var(der)
    high = set_var(highp)
    low = set_var(lowp)
    close = set_var(closep)
    vol = set_var(volume)
    
    # Return as single items for easy analysis
    inputs = []
    inputs.append([ema7, "EMA7"])
    inputs.append([ema50, "EMA50"])
    inputs.append([ema200, "EMA200"])
    inputs.append([rsi, "RSI"])
    inputs.append([adx, "ADX"])
    inputs.append([macd, "MACD"])
    inputs.append([signal, "Signal"])
    inputs.append([der, "Derivative"])
    inputs.append([high, "High"])
    inputs.append([low, "Low"])
    inputs.append([close, "Close"])
    inputs.append([vol, "Volume"])
    
    outputs = close
    
    return inputs, outputs
    
    