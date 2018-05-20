import numpy as np
import cv2
import os
import sys
#import matplotlib.pyplot as plt
import scipy.ndimage.filters as F
import scipy.stats as ST
from scipy import signal,ndimage
import glob

SCALE = 1
RANK = 300
RADIUS = 3
S_DIV = 7
DIV = 16
MIN_LOSS = 500
eps = 1e-10

# file read
temp_files = glob.glob('template/*.ppm')
back_files = glob.glob('ppm/*.ppm')
temp_imgs = {os.path.basename(t):cv2.imread(t) for t in temp_files}
back_imgs = {os.path.basename(c):cv2.imread(c) for c in back_files}

# lambda fanc
rot_xy = lambda r: np.array([[np.cos(r), np.sin(r)],[np.sin(r), np.cos(r)]])

#def fanc

def grad_2d(img):
    if len(img.shape)==3:
        img = np.mean(img, axis=2)
    img[img==0] = 127.5
    x = np.gradient(img,axis=0)
    y = np.gradient(img,axis=1)
    return np.hypot(x,y)

def img_affine(img, center, deg, scale=1.):
    cy, cx = center
    matrix = cv2.getRotationMatrix2D((cx,cy), deg, scale)
    img_sp = cv2.split(img)
    res = [cv2.warpAffine(p,matrix,(p.shape[1],p.shape[0]),flags=cv2.INTER_CUBIC) for p in img_sp]
    return np.dstack(res)
    

def blank_to_zero(img):
    img[np.all(img==255, axis=2)]=0
    return img

def gaussian_kernel(klen=21, sig=3):
    interval = (2*sig+1.)/(klen)
    x = np.linspace(-sig-interval/2., sig+interval/2., klen+1)
    kern1d = np.diff(ST.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def hist_rgb(img):
    rgb = cv2.split(img)
    hists = (np.histogram(x.ravel(),255,[1,255])[0] for x in rgb)
    return np.dstack(hists)

def get_kp_dog(img):
    #init
    init_sig = 1.1; div=S_DIV
    k=np.power(2,1/div)
    #to grayscale
    gray_img = np.mean(img, axis=2)

    #generate sigma list
    sig_li = np.logspace(0,div,div,base=k)*init_sig
    g_imgs = [F.gaussian_filter(gray_img, s) for s in sig_li]
    dog_imgs = np.dstack(np.diff(g_imgs, axis=0))
    h,w = dog_imgs.shape[:2]
    kp_li = []
    for y in range(h-2):
        for x in range(w-2):
            if gray_img[y,x]==0: continue
            if np.mean(np.abs(dog_imgs[y,x]))<0.2: continue
            vals = dog_imgs[y:y+3,x:x+3,:]
            center = np.squeeze(dog_imgs[y+1, x+1, 1:div-1])
            max_ = np.max(np.abs(vals))
            condition = np.where(np.abs(center)==max_)
            if condition[0].any():
                kp_li.append(np.array([y,x,condition[0][0]]))
    return kp_li, sig_li

def kp_writer(b_img, points, cogs, name='cogs'):
    back = b_img
    h,w,c = np.shape(b_img)
    rescale = 3
    back = cv2.resize(back, None, fx=rescale, fy=rescale)
    for p,c in zip(points, cogs):
        y,x,s = p
        cy, cx = c
        r = 5; l = 10; d=int((S_DIV-2)/5)
        if s<d: cv2.circle(back, (x*3,y*3), int((s+1)*r), (0,0,255), 1)
        elif s<d*2: cv2.circle(back, (x*3,y*3), int((s+1)*r), (0,255,255), 1)
        elif s<d*3: cv2.circle(back, (x*3,y*3), int((s+1)*r), (0,255,0), 1)
        elif s<d*4: cv2.circle(back, (x*3,y*3), int((s+1)*r), (255,255,0), 1)
        else: cv2.circle(back, (x*3,y*3), int((s+1)*r), (255,0,0), 1)
        if 0<y+cy<h and 0<x+cx<w:
            cv2.line(back, (x*3,y*3), (int(x+c[1]*l)*3,int(y+c[0]*l)*3), (0,0,255), 1)
    cv2.imwrite('result/'+name+'.ppm',back)

def match_writer(a_img, b_img, pair_li):
    h,w,c = np.shape(a_img)
    h_,w_,c_ = np.shape(b_img)
    pad_a = np.pad(a_img, [(0,h_-h),(0,w_-w),(0,0)], 'constant')
    concat = np.concatenate([pad_a, b_img], axis=1)
    for p in pair_li:
        cv2.line(concat, (p[0][1], p[0][0]), (p[1][1]+w_, p[1][0]), (255,0,255), 1)
    cv2.imwrite('result/concat.ppm', concat)


def cal_cog(matrix): 
    h,w = matrix.shape
    d_mat = np.arange(w, dtype=float) - (w-1.)/2.
    d_x = np.tile(d_mat, [h,1])
    m_sum = np.sum(matrix)
    x = np.sum(matrix*d_x)/(m_sum+eps)
    y = np.sum(matrix*d_x.T)/(m_sum+eps)
    return [y,x]

def get_cog_grad(img, points, sigs):
    cogs = []
    for p in points:
        y,x,s = p
        sig = sigs[s]
        n = np.int(np.trunc(sig*4))
        a = np.int(n//2)
        spot = np.pad(img,[(a,a),(a,a),(0,0)],'constant')[y:y+n,x:x+n,:]
        grad = grad_2d(spot)
        grad_weighted = np.multiply(gaussian_kernel(klen=grad.shape[0],sig=sig),grad)
        cog = cal_cog(np.squeeze(grad_weighted))
        cogs.append(cog)
    return cogs

# histgram matching
def hist_mat(a_img, b_img, scale, stride=1, ocl_param=100):
    # init
    a_img_g = np.dstack([F.gaussian_filter(a,1) for a in cv2.split(a_img)])
    b_img_g = np.dstack([F.gaussian_filter(b,1) for b in cv2.split(b_img)])
    a_hist = hist_rgb(a_img_g)
    a_h, a_w = np.shape(a_img)[:2]
    b_h, b_w = np.shape(b_img)[:2]
    scale_h, scale_w = map(int,(a_h*scale, a_w*scale))
    max_s = 0
    max_x, max_y = (0, 0)
    re_x, re_y = (0, 0)

    # fit
    for y in range(0,b_h-scale_h,stride):
        for x in range(0,b_w-scale_w,stride):
            trim = b_img_g[y:y+scale_h,x:x+scale_w]
            b_hist = hist_rgb(trim)
            score = np.sum(np.minimum(a_hist,b_hist))
            if max_s < score+ocl_param:
                re_x, re_y = x, y
                if max_s < score:
                    max_s = score
                    max_x, max_y = x, y
    cen_x, cen_y = (int((re_x+max_x)/2), int((re_y+max_y)/2))

    # result
    extra_img = b_img[cen_y:cen_y+scale_h,cen_x:cen_x+scale_w]
    return extra_img, (cen_y, cen_x), (scale_h, scale_w) 

def get_rgb_feat(img, center, vec, sig):
    cy,cx = center
    rad = np.arctan(center[1]/(center[0]+eps))
    img = img_affine(img, center, np.rad2deg(rad))
    n = np.int(sig*9)
    sn = np.int(sig*3)
    p = np.int(n//2)
    pad_img = np.pad(img,[(p,n),(p,n),(0,0)],'constant')
    spot = pad_img[cy:cy+n,cx:cx+n,:]
    hist = hist_rgb(spot)
    spot_weighted = np.multiply(np.expand_dims(gaussian_kernel(klen=spot.shape[0],sig=sig),axis=2),spot)
    grid = np.resize(np.squeeze(spot_weighted), (sn,sn,3*3,3))
    f_vec = [[cal_cog(grid[:,:,i,c]) for c in range(3)] for i in range(grid.shape[2])]
    return np.array(f_vec), hist
    

def kp_mat(a_img, b_img, a_ps, b_ps, a_cogs, b_cogs, sigs):
    match_pair = []
    a_img = np.dstack([F.gaussian_filter(a,1) for a in cv2.split(a_img)])
    b_img = np.dstack([F.gaussian_filter(b,1) for b in cv2.split(b_img)])
    for i,(a_p,a_cog) in enumerate(zip(a_ps,a_cogs)):
        a_vec, a_hist = get_rgb_feat(a_img, (a_p[0],a_p[1]), a_cog, sigs[a_p[2]])
        if np.max(a_vec)<0.1: continue
        min_err = np.inf; min_j=0
        for j,(b_p,b_cog) in enumerate(zip(b_ps,b_cogs)):
            if i==j : continue
            b_vec, b_hist = get_rgb_feat(b_img, (b_p[0],b_p[1]), b_cog, sigs[b_p[2]])
            if np.max(b_vec)<0.1: continue
            err = np.linalg.norm(a_vec-b_vec) + 1.-np.sum(np.minimum(b_hist,a_hist))/np.sum(np.maximum(a_hist,b_hist))
            if err<min_err:
                min_err=err; min_j=j
        item = {'err':min_err, 'i':i, 'j':min_j}
        print(min_err)
        match_pair.append(item)
    res_pair = sorted(match_pair, key=lambda x:x['err'])
    return res_pair


# main-code

## command check
if(len(sys.argv) != 3):
    print('Not found 2 filenames:(ex: python3 fit.py inu.ppm class1_b2_n50_1.ppm)')
    exit()

## load image
temp_name = sys.argv[1]
back_name = sys.argv[2]
temp_img = blank_to_zero(temp_imgs[temp_name])
#cv2.imwrite('result/temp_img_g.ppm', img_affine(temp_img, (30,100), 10))
back_img = back_imgs[back_name]
t_h, t_w = np.shape(temp_img)[:2]
b_h, b_w = np.shape(back_img)[:2]
print('temp_size:(%d,%d)'%(t_h,t_w))
print('image_size:(%d,%d)'%(b_h,b_w))
#plt.plot(temp_hist[0])
a_img = temp_img
b_img = back_img

## hist
trim_img, cen_yx, size = hist_mat(a_img, b_img, stride=10, scale=SCALE, ocl_param=1000)
c_y, c_x = cen_yx[0], cen_yx[1]; s_h, s_w = size[0], size[1]
#cv2.imwrite('result/hist_res_img.ppm', trim_img)

## temp keypoint
a_kp_li, a_sigs = get_kp_dog(a_img)
print('Key points [temp]:',len(a_kp_li))
a_cog_li = get_cog_grad(a_img, a_kp_li, a_sigs)
kp_writer(a_img, a_kp_li, a_cog_li, name='a_kp')

## temp keypoint
b_kp_li, b_sigs = get_kp_dog(b_img)
print('Key points [back]:',len(b_kp_li))
b_cog_li = get_cog_grad(b_img, b_kp_li, b_sigs)
kp_writer(b_img, b_kp_li, b_cog_li, name='b_kp')

## trim keypoint
t_kp_li, t_sigs = get_kp_dog(trim_img)
print('Key points [trim]:',len(t_kp_li))
t_cog_li = get_cog_grad(trim_img, t_kp_li, t_sigs)
kp_writer(trim_img, t_kp_li, t_cog_li, name='t_kp')

## matching
mat_dict = kp_mat(a_img, b_img, a_kp_li, b_kp_li, a_cog_li, b_cog_li, a_sigs)
pair_li = [[a_kp_li[m['i']], b_kp_li[m['j']]] for m in mat_dict[:10]]
match_writer(a_img, b_img, pair_li)

exit()
