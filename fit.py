import numpy as np
import cv2
import os
import sys
#import matplotlib.pyplot as plt
from scipy import signal,ndimage
import glob

SCALE = 1
RANK = 100
RADIUS = 3
DIV = 16
MIN_LOSS = 100

# file read
temp_files = glob.glob('template/*.ppm')
back_files = glob.glob('ppm/*.ppm')
temp_imgs = {os.path.basename(t):cv2.imread(t) for t in temp_files}
back_imgs = {os.path.basename(c):cv2.imread(c) for c in back_files}

# filter
fil_gaus = np.array([[1/16, 1/8, 1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
fil_y = np.array([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]])
fil_x = fil_y.transpose()

# lambda fanc
norm = lambda v: v/np.sum(v)
argmax_2d = lambda x: list(zip(*np.unravel_index(np.argsort(x.ravel())[::-1],x.shape)))
f_gaus = lambda img: np.dstack(signal.convolve2d(p, fil_gaus, 'same') for p in cv2.split(img))

#def fanc
def blank_to_zero(img):
    img[np.all(img==255, axis=2)]=0
    return img

def get_fp(img, rank, rad=RADIUS):
    h, w = img.shape[:2]
    img = conv_rgb(img)
    arg_list = argmax_2d(img[:,:,0]+img[:,:,1]+img[:,:,2])[:rank]
    
    return arg_list, img


def hist_rgb(img):
    rgb = cv2.split(img)
    hists = (np.histogram(x.ravel(),255,[1,255])[0] for x in rgb)
    return np.dstack(hists)

def conv_rgb(img):
    rgb = cv2.split(img)
    rgb = (signal.convolve2d(p, fil_gaus, 'same') for p in rgb)
    rgb = (signal.convolve2d(p, fil_y, 'same') for p in rgb)
    res = (signal.convolve2d(p, fil_x, 'same') for p in rgb)
    return np.dstack(res)

def fp_writer(arg_list,shape, name='feature_point'):
    point_img = np.zeros(shape)
    for args in arg_list: point_img[args] = 255
    
    cv2.imwrite('result/'+name+'.ppm',point_img)
    return point_img

def make_feat_data(f_map, points, rad=RADIUS, scale=1):
    div=DIV; spec=10;
    coords = []
    rad*=scale
    for i in range(div):
        r = 2*np.pi/div
        y = int(np.round(rad * np.sin(r*i)))
        x = int(np.round(rad * np.cos(r*i)))
        coords.append((y,x))
        #print(y,x)
    d0, d1, d2 = f_map.shape
    point_val = np.zeros((d0, d1, div, d2))
    new_p_li = []
    for p in points:
        py, px = p
        if py<rad or px<rad or py+rad>=d0 or px+rad>=d1: continue
        arr = np.array([f_map[py+c[0],px+c[1]] for c in coords])
        if np.sum(arr>127)>spec:
            point_val[p] = arr
            if p not in new_p_li: new_p_li.append(p)
    return point_val, new_p_li

def sway(data, d, div=DIV):
    if d==0: return data
    elif d<0 :
        data=data[::-1]
        dd=abs(d)
    else: dd=d
    roll_i, rem = divmod(dd,(360/div))
    data = np.roll(data,-int(roll_i),axis=0)
    d_deg = rem/(360/div)
    data = np.append(data,[data[0]],axis=0)
    data = np.array([(1-d_deg)*dat+d_deg*data[i+1] for i,dat in enumerate(data[:-1])])
    if d<0 : return data[::-1]
    else : return data

def loss_cal(a,b):
    map(np.array,(a,b))
    if np.sum(b)-np.sum(a)>50: return np.inf, 0
    li = []
    deg_li = [i for i in range(-30,30+1)]
    for i in deg_li:
        b_s = sway(b,-i)
        b_s[b_s==0] = a[b_s==0]
        li.append(np.linalg.norm(a-b_s)) 
    arg = np.argmin(li)
    return li[arg], deg_li[arg]



# histgram matching
def hist_mat(a_img, b_img, scale, stride=10, ocl_param=100):
    # init
    a_hist = hist_rgb(a_img)
    a_h, a_w = np.shape(a_img)[:2]
    b_h, b_w = np.shape(b_img)[:2]
    scale_h, scale_w = map(int,(a_h*scale, a_w*scale))
    max_s = 0
    max_x, max_y = (0, 0)
    re_x, re_y = (0, 0)

    # fit
    for y in range(0,b_h-scale_h,stride):
        for x in range(0,b_w-scale_w,stride):
            trim = b_img[y:y+scale_h,x:x+scale_w]
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


# feature matching
def feat_mat(a_img, b_img, rank=RANK, scale=1):
    # init
    V1_NUM = 100

    # temp
    feat_a, conv_a = get_fp(a_img, rank=rank)
    ap_img = fp_writer(feat_a, a_img.shape, name='temp_feature')
    a_val, feat_a = make_feat_data(f_gaus(a_img), feat_a)
    cv2.imwrite('result/val_img.ppm', np.sum(a_val, axis=2))

    # back_image
    conv_b = conv_rgb(b_img)

    min_err = np.inf
    vec_list = []

    feat_b_paired = []
    for ap in feat_a[:V1_NUM]: 
        for y in range(b_img.shape[0]):
            for x in range(b_img.shape[1]):
                bp = (y,x)
                err = np.sum(np.abs(conv_b[y,x]-conv_a[ap]))
                if err < 100 : 
                    feat_b_paired.append(bp)
    
    b_val, feat_b = make_feat_data(f_gaus(b_img), feat_b_paired, scale=scale)
    cv2.imwrite('result/b_val_img.ppm', np.sum(b_val, axis=2))

    min_loss = MIN_LOSS
    eps = 1e-10
    for ap in feat_a[:V1_NUM]:
        for bp in feat_b:
            loss, deg = loss_cal(a_val[ap], b_val[bp])
            if loss < MIN_LOSS:
                vec_list.append({'a':np.array(ap), 'b':np.array(bp), 'loss':loss, 'deg':deg})
    mo = (0,0)
    if len(vec_list)<=2: return (mo, 0, 0)

    vec_list = sorted(vec_list, key=lambda x:x['loss']) 
    f_vec_li = vec_list
    for v in vec_list[:20]:
        print("[add_vec] loss:%d deg:%f"%(v['loss'],v['deg']),v['a'],v['b'])
    d_vec_li = []; vec_deg_li = []

    for f_vec in f_vec_li[:10]:
        va = np.array(f_vec['b']); vb = np.array(f_vec['a'])
        d_vec_li.append(vb-va)
        vec_deg_li.append(f_vec['deg'])
    npint = np.frompyfunc(int,1,1)
    d_vec = npint(np.mean(d_vec_li, axis=0))
    vec_deg = np.int(np.mean(vec_deg_li))
    vec_scale = 1.
    return d_vec, vec_scale, vec_deg

# main-code
## load image
if(len(sys.argv) != 3):
    print('Not found 2 filenames:(ex: python3 fit.py inu.ppm class1_b2_n50_1.ppm)')
    exit()

temp_name = sys.argv[1]
back_name = sys.argv[2]

temp_img = blank_to_zero(temp_imgs[temp_name])
back_img = back_imgs[back_name]
t_h, t_w = np.shape(temp_img)[:2]
b_h, b_w = np.shape(back_img)[:2]

print('temp_size:(%d,%d)'%(t_h,t_w))
print('image_size:(%d,%d)'%(b_h,b_w))

#plt.plot(temp_hist[0])

a_img = temp_img
b_img = back_img
trim_img, cen_yx, size = hist_mat(a_img, b_img, stride=1, scale=SCALE)
c_y, c_x = cen_yx[0], cen_yx[1]; s_h, s_w = size[0], size[1]
cv2.imwrite('result/hist_res_img.ppm', trim_img)

move, rescale, redeg = feat_mat(a_img, trim_img, scale=SCALE)
print('move:(',move[0],',',move[1],') scale:',rescale,' deg:',redeg)
c_y -= move[0]
c_x -= move[1]

cv2.rectangle(b_img, (c_x,c_y), (c_x+s_w,c_y+s_h), (0,0,255), 2)
cv2.imwrite('result/res_img.ppm', b_img)

#rotation & resize
#fix_img = ndimage.rotate(fix_img, np.rad2deg(redeg), reshape=False)
#res_img = cv2.resize(fix_img, None, fx=rescale, fy=rescale)


print("[finish] matching result : ",c_x+int(s_w/2),c_y+int(s_h/2))
