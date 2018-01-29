import numpy as np

np.random.seed(56789)

def ReLU(x):
    mask = (x > 0) * 1.0
    return x * mask
def d_ReLU(x):
    mask = (x > 0) * 1.0
    return mask

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def log(x):
    return 1 / (1 + np.exp(-1 *x))
def d_log(x):
    return log(x) * (1 - log(x))

# 1. Declare training data and hyper parameter
num_epoch = 5000
learing_rate = 0.01
learing_rate_rec = 0.0001

x = np.array([
    [0,0.5,0],
    [0.4,-0.2,0.2],
    [0.1,0.2,0.2]    
])

y = np.array([
    [0,0.5,0.5],
    [0.4,0.2,0.4],
    [0.1,0.3,0.5]    
])

h = np.zeros((x.shape[0],x.shape[1] + 1))

wrecyy,wxyy = np.random.randn() * 0.2,np.random.randn() * 0.2
wrechh,wxhh = np.random.randn() * 0.2,np.random.randn() * 0.2
wrecgy,wxgy = np.random.randn() * 0.2,np.random.randn() * 0.2
wrecgh,wxgh = np.random.randn() * 0.2,np.random.randn() * 0.2


# ---- Forward Feed at TS = 1 -------
yy1 = wrecyy * h[:,0] + wxyy * x[:,0]
yy1A = ReLU(yy1)

hh1 = wrechh * h[:,0] + wxhh * x[:,0]
hh1A = tanh(hh1)

gy1 = wrecgy * h[:,0] + wxgy * x[:,0]
gy1A = log(gy1)

gh1 = wrecgh * h[:,0] + wxgh * x[:,0]
gh1A = log(gh1)

y1 = gy1A * x[:,0] + ( 1-gy1A ) * yy1A
h[:,1] = gh1A * h[:,0] + ( 1-gh1A ) * hh1A

# ---- Forward Feed at TS = 2 -------
yy2 = wrecyy * h[:,1] + wxyy * x[:,1]
yy2A = ReLU(yy2)

hh2 = wrechh * h[:,1] + wxhh * x[:,1]
hh2A = tanh(hh2)

gy2 = wrecgy * h[:,1] + wxgy * x[:,1]
gy2A = log(gy2)

gh2 = wrecgh * h[:,1] + wxgh * x[:,1]
gh2A = log(gh2)

y2 = gy2A * x[:,1] + ( 1-gy2A ) * yy2A
h[:,2] = gh2A * h[:,1] + ( 1-gh2A ) * hh2A

# ---- Forward Feed at TS = 3 -------
yy3 = wrecyy * h[:,2] + wxyy * x[:,2]
yy3A = ReLU(yy3)

hh3 = wrechh * h[:,2] + wxhh * x[:,2]
hh3A = tanh(hh3)

gy3 = wrecgy * h[:,2] + wxgy * x[:,2]
gy3A = log(gy3)

gh3 = wrecgh * h[:,2] + wxgh * x[:,2]
gh3A = log(gh3)

y3 = gy3A * x[:,2] + ( 1-gy3A ) * yy3A
h[:,3] = gh3A * h[:,2] + ( 1-gh3A ) * hh3A


print('----- Ground Truth ------')
print(y)
print('----- Before Train Predict Y  ------')
print(y1,'\n',y2,'\n',y3)
print('----- Before Train Predict H ------')
print(h[:,1],'\n',h[:,2],'\n',h[:,3],'\n\n')


for iter in range(num_epoch):

    # ---- Forward Feed at TS = 1 -------
    yy1 = wrecyy * h[:,0] + wxyy * x[:,0]
    yy1A = ReLU(yy1)

    hh1 = wrechh * h[:,0] + wxhh * x[:,0]
    hh1A = tanh(hh1)

    gy1 = wrecgy * h[:,0] + wxgy * x[:,0]
    gy1A = log(gy1)

    gh1 = wrecgh * h[:,0] + wxgh * x[:,0]
    gh1A = log(gh1)

    y1 = gy1A * x[:,0] + ( 1-gy1A ) * yy1A
    h[:,1] = gh1A * h[:,0] + ( 1-gh1A ) * hh1A
    
    # ---- Forward Feed at TS = 2 -------
    yy2 = wrecyy * h[:,1] + wxyy * x[:,1]
    yy2A = ReLU(yy2)

    hh2 = wrechh * h[:,1] + wxhh * x[:,1]
    hh2A = tanh(hh2)

    gy2 = wrecgy * h[:,1] + wxgy * x[:,1]
    gy2A = log(gy2)

    gh2 = wrecgh * h[:,1] + wxgh * x[:,1]
    gh2A = log(gh2)

    y2 = gy2A * x[:,1] + ( 1-gy2A ) * yy2A
    h[:,2] = gh2A * h[:,1] + ( 1-gh2A ) * hh2A

    # ---- Forward Feed at TS = 3 -------
    yy3 = wrecyy * h[:,2] + wxyy * x[:,2]
    yy3A = ReLU(yy3)

    hh3 = wrechh * h[:,2] + wxhh * x[:,2]
    hh3A = tanh(hh3)

    gy3 = wrecgy * h[:,2] + wxgy * x[:,2]
    gy3A = log(gy3)

    gh3 = wrecgh * h[:,2] + wxgh * x[:,2]
    gh3A = log(gh3)

    y3 = gy3A * x[:,2] + ( 1-gy3A ) * yy3A
    h[:,3] = gh3A * h[:,2] + ( 1-gh3A ) * hh3A

    cost_y1,cost_h1 = np.square(y1-y[:,0]).sum() * 0.5,np.square(h[:,1]-y[:,0] ).sum() * 0.5
    cost_y2,cost_h2 = np.square(y2-y[:,1]).sum() * 0.5,np.square(h[:,2]-y[:,1] ).sum() * 0.5
    cost_y3,cost_h3 = np.square(y3-y[:,2]).sum() * 0.5,np.square(h[:,3]-y[:,2] ).sum() * 0.5

    total_cost_y = cost_y1 + cost_y2 + cost_y3
    total_cost_h = cost_h1 + cost_h2 + cost_h3

    print("Current iter: ", iter, ' Current Y Total Cost: ', total_cost_y,' Current h Total Cost: ', total_cost_h,end='\r')
    

    grad_common_ts_3_yy = (y3-y[:,2]) * ( 1-gy3A )
    grad_common_ts_3_hh = (h[:,3]-y[:,2]) * ( 1-gh3A )
    grad_common_ts_3_gy = (y3-y[:,2]) * (x[:,2]  - yy3A)
    grad_common_ts_3_gh = (h[:,3]-y[:,2]) * (h[:,2]  - hh3A)


    grad_common_ts_2_yy = (y2-y[:,1]) * ( 1-gy2A )
    grad_common_ts_2_hh = (h[:,2]-y[:,1]) * ( 1-gh2A ) + \
        (h[:,3]-y[:,2]) * ((1-gh3A) * (d_tanh(hh3)) * (wrechh) + (gh3A) + (h[:,2] -hh3A) * (d_log(gh3)) * (wrecgh)) * (1-gh2A)

    grad_common_ts_2_gy = (y2-y[:,1]) * (x[:,1]  - yy2A)
    grad_common_ts_2_gh = (h[:,2]-y[:,1]) * (h[:,1]  - hh2A) + \
        (h[:,3]-y[:,2]) * ((1-gh3A) * (d_tanh(hh3)) * (wrechh) + (gh3A) + (h[:,2] -hh3A) * (d_log(gh3)) * (wrecgh)) * (h[:,1]  - hh2A)


    grad_common_ts_1_yy = (y1-y[:,0]) * ( 1-gy1A )
    grad_common_ts_1_hh = (h[:,1]-y[:,0]) * ( 1-gh1A ) + \
        (h[:,2]-y[:,1]) * ((1-gh2A) * (d_tanh(hh2)) * (wrechh) + (gh2A) + (h[:,1] -hh2A) * (d_log(gh2)) * (wrecgh)) * (1-gh1A) +  \
        (h[:,3]-y[:,2]) * (
            (1-gh3A) * (d_tanh(hh3)) * (wrechh) * ( (1-gh2A) * (d_tanh(hh2)) * (wrechh) + (gh2A) + (h[:,1] -hh2A) * (d_log(gh2)) * (wrecgh) )   + 
            (gh3A) * ((1-gh2A) *  (d_tanh(hh2)) * (wrechh) + (gh2A) + (h[:,1] -hh2A) * (d_log(gh2)) * (wrecgh))  + 
            (h[:,2] -hh3A) * (d_log(gh3)) * (wrecgh) * ((1-gh2A) *  (d_tanh(hh2)) * (wrechh) + (gh2A) + (h[:,1] -hh2A) * (d_log(gh2)) * (wrecgh))              
        ) * (1 - gh1A)

    grad_common_ts_1_gy = (y1-y[:,0]) * (x[:,0]  - yy1A)
    grad_common_ts_1_gh = (h[:,1]-y[:,0]) * (h[:,0]  - hh1A) + \
        (h[:,2]-y[:,1]) * ((1-gh2A) * (d_tanh(hh2)) * (wrechh) + (gh2A) + (h[:,1] -hh2A) * (d_log(gh2)) * (wrecgh)) * (h[:,0]  - hh1A) +  \
        (h[:,3]-y[:,2]) * (
            (1-gh3A) * (d_tanh(hh3)) * (wrechh) * ( (1-gh2A) * (d_tanh(hh2)) * (wrechh) + (gh2A) + (h[:,1] -hh2A) * (d_log(gh2)) * (wrecgh) )   + 
            (gh3A) * ((1-gh2A) *  (d_tanh(hh2)) * (wrechh) + (gh2A) + (h[:,1] -hh2A) * (d_log(gh2)) * (wrecgh))  + 
            (h[:,2] -hh3A) * (d_log(gh3)) * (wrecgh) * ((1-gh2A) *  (d_tanh(hh2)) * (wrechh) + (gh2A) + (h[:,1] -hh2A) * (d_log(gh2)) * (wrecgh))              
        ) * (h[:,0]  - hh1A)
    
    
    grad_wrecyy = np.sum(
        grad_common_ts_3_yy * (d_ReLU(yy3)) * h[:,2] + 
        grad_common_ts_2_yy * (d_ReLU(yy2)) * h[:,1] + 
        grad_common_ts_1_yy * (d_ReLU(yy1)) * h[:,0] 
    )
    grad_wxyy = np.sum(
        grad_common_ts_3_yy * (d_ReLU(yy3)) * x[:,2] + 
        grad_common_ts_2_yy * (d_ReLU(yy2)) * x[:,1] + 
        grad_common_ts_1_yy * (d_ReLU(yy1)) * x[:,0] 
    )

    grad_wrechh = np.sum(
        grad_common_ts_3_hh * (d_tanh(hh3)) * h[:,2] + 
        grad_common_ts_2_hh * (d_tanh(hh2)) * h[:,1] + 
        grad_common_ts_1_hh * (d_tanh(hh1)) * h[:,0] 
    )
    grad_wxhh = np.sum(
        grad_common_ts_3_hh * (d_tanh(hh3)) * x[:,2] + 
        grad_common_ts_2_hh * (d_tanh(hh2)) * x[:,1] + 
        grad_common_ts_1_hh * (d_tanh(hh1)) * x[:,0] 
    )

    grad_wrecgy = np.sum(
        grad_common_ts_3_gy * (d_log(gy3)) * h[:,2] + 
        grad_common_ts_2_gy * (d_log(gy2)) * h[:,1] + 
        grad_common_ts_1_gy * (d_log(gy1)) * h[:,0] 
    )

    grad_wxgy = np.sum(
        grad_common_ts_3_gy * (d_log(gy3)) * x[:,2] + 
        grad_common_ts_2_gy * (d_log(gy2)) * x[:,1] + 
        grad_common_ts_1_gy * (d_log(gy1)) * x[:,0] 
    )

    grad_wrecgh = np.sum(
        grad_common_ts_3_gh * (d_log(gh3)) * h[:,2] + 
        grad_common_ts_2_gh * (d_log(gh2)) * h[:,1] + 
        grad_common_ts_1_gh * (d_log(gh1)) * h[:,0] 
    )
    grad_wxgh = np.sum(
        grad_common_ts_3_gh * (d_log(gh3)) * x[:,2] + 
        grad_common_ts_2_gh * (d_log(gh2)) * x[:,1] + 
        grad_common_ts_1_gh * (d_log(gh1)) * x[:,0] 
    ) 
    
    wrecyy = wrecyy - learing_rate_rec *grad_wrecyy
    wxyy = wxyy - learing_rate*grad_wxyy

    wrechh=wrechh- learing_rate_rec*grad_wrechh
    wxhh =wxhh- learing_rate*grad_wxhh

    wrecgy=wrecgy- learing_rate_rec*grad_wrecgy
    wxgy =wxgy- learing_rate*grad_wxgy

    wrecgh=wrecgh- learing_rate_rec*grad_wrecgh
    wxgh =wxgh - learing_rate*grad_wxgh


# ---- Forward Feed at TS = 1 -------
yy1 = wrecyy * h[:,0] + wxyy * x[:,0]
yy1A = ReLU(yy1)

hh1 = wrechh * h[:,0] + wxhh * x[:,0]
hh1A = tanh(hh1)

gy1 = wrecgy * h[:,0] + wxgy * x[:,0]
gy1A = log(gy1)

gh1 = wrecgh * h[:,0] + wxgh * x[:,0]
gh1A = log(gh1)

y1 = gy1A * x[:,0] + ( 1-gy1A ) * yy1A
h[:,1] = gh1A * h[:,0] + ( 1-gh1A ) * hh1A

# ---- Forward Feed at TS = 2 -------
yy2 = wrecyy * h[:,1] + wxyy * x[:,1]
yy2A = ReLU(yy2)

hh2 = wrechh * h[:,1] + wxhh * x[:,1]
hh2A = tanh(hh2)

gy2 = wrecgy * h[:,1] + wxgy * x[:,1]
gy2A = log(gy2)

gh2 = wrecgh * h[:,1] + wxgh * x[:,1]
gh2A = log(gh2)

y2 = gy2A * x[:,1] + ( 1-gy2A ) * yy2A
h[:,2] = gh2A * h[:,1] + ( 1-gh2A ) * hh2A

# ---- Forward Feed at TS = 3 -------
yy3 = wrecyy * h[:,2] + wxyy * x[:,2]
yy3A = ReLU(yy3)

hh3 = wrechh * h[:,2] + wxhh * x[:,2]
hh3A = tanh(hh3)

gy3 = wrecgy * h[:,2] + wxgy * x[:,2]
gy3A = log(gy3)

gh3 = wrecgh * h[:,2] + wxgh * x[:,2]
gh3A = log(gh3)

y3 = gy3A * x[:,2] + ( 1-gy3A ) * yy3A
h[:,3] = gh3A * h[:,2] + ( 1-gh3A ) * hh3A


print('\n\n----- Ground Truth ------')
print(y)
print('----- After Train Predict Y  ------')
print(y1,'\n',y2,'\n',y3)
print('----- After Train Predict H ------')
print(h[:,1],'\n',h[:,2],'\n',h[:,3])



# ---- end code ---   
