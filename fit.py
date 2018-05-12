import numpy as np
import cv2
#import matplotlib.pyplot as plt
from scipy import signal,ndimage
import glob

RANK = 500
RADIUS = 3
MIN_LOSS = 100

# file read
temp_files = glob.glob('template/*.ppm')
back_files = glob.glob('ppm/*.ppm')
temp_imgs = {t.split('/')[-1]:cv2.imread(t) for t in temp_files}
back_imgs = {c.split('/')[-1]:cv2.imread(c) for c in back_files}

# filter
fil_y = np.array([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]])
fil_x = fil_y.transpose()

# lambda fanc
norm = lambda v: v/np.sum(v)
argmax_2d = lambda x: list(zip(*np.unravel_index(np.argsort(x.ravel())[::-1],x.shape)))

#def fanc
def blank_to_zero(img):
    img[np.all(img==255, axis=2)]=0
    return img


def get_fp(img, rank, rad=RADIUS):
    fp = []
    h, w = img.shape[:2]
    rad_area = np.zeros(img.shape)
    img = conv_rgb(img)
    rgb = cv2.split(img)
    for i,p in enumerate(rgb):
        arg_list = argmax_2d(p)[:rank]
        [fp.append((arg[0], arg[1], i)) for arg in arg_list]
    return fp, img


def hist_rgb(img):
    rgb = cv2.split(img)
    hists = (np.histogram(x.ravel(),255,[1,255])[0] for x in rgb)
    return np.dstack(hists)

def conv_rgb(img):
    rgb = cv2.split(img)
    rgb = (signal.convolve2d(p, fil_y, 'same') for p in rgb)
    res = (signal.convolve2d(p, fil_x, 'same') for p in rgb)
    return np.dstack(res)

def fp_writer(arg_list,shape, name='feature_point'):
    point_img = np.zeros(shape)
    for args in arg_list: point_img[args] = 255
    
    cv2.imwrite('result/'+name+'.ppm',point_img)
    return point_img

def make_feat_data(f_map, points, rad=RADIUS, scale=1):
    div=16; spec=4;
    coords = []
    rad*=scale
    for i in range(div):
        r = 2*np.pi/div
        y = int(np.round(rad * np.sin(r*i)))
        x = int(np.round(rad * np.cos(r*i)))
        coords.append((y,x))
        #print(y,x)

    d0, d1, d2 = f_map.shape
    point_val = np.zeros([d0, d1, d2, div])
    new_p_li = []
    for p in points:
        py,px,pc = p
        if py<rad or px<rad or py+rad>=d0 or px+rad>=d1: continue
        arr = np.array([f_map[py+c[0],px+c[1],pc] for c in coords])
        if np.abs(np.sum(arr>127)-int(div/2))<spec:
            point_val[p] = arr
            new_p_li.append(p)
    return point_val, new_p_li

def loss_cal(a,b):
    map(np.array,(a,b))
    if np.sum(b)-np.sum(a)>50: return np.inf, 0
    li = []
    for i in range(len(b)):
        b = np.roll(b,i)
        b[b==0] = a[b==0]
        li.append(np.linalg.norm(a-b)) 
    arg = np.argmin(li)
    return li[arg], arg-1



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
    conv_a = conv_rgb(a_img) 
    feat_a = argmax_2d(conv_a)[:rank]
    ap_img = fp_writer(feat_a, a_img.shape, name='temp_feature')
    a_val, feat_a = make_feat_data(a_img, feat_a)
    cv2.imwrite('result/val_img.ppm', np.sum(a_val, axis=3))

    # back_image
    conv_b = conv_rgb(b_img)

    min_err = np.inf
    vec_list = []

    feat_b_paired = []
    for ap in feat_a[:V1_NUM]: 
        for y in range(b_img.shape[0]):
            for x in range(b_img.shape[1]):
                err = np.sum(np.abs(conv_b[y,x]-conv_a[ap[:2]]))
                if err < 300:
                    feat_b_paired.append((y,x,ap[2]))
    
    b_val, feat_b = make_feat_data(b_img, feat_b_paired, scale=scale)
    cv2.imwrite('result/b_val_img.ppm', np.sum(b_val, axis=3))

    min_loss = MIN_LOSS
    eps = 1e-10
    for ap in feat_a[:V1_NUM]:
        for bp in feat_b:
            if ap[2] != bp[2]: continue
            loss, deg = loss_cal(a_val[ap], b_val[bp])
            if min_loss>=loss:
                vec_list.append({'a':np.array(ap), 'b':np.array(bp), 'loss':loss, 'deg':deg})
                print("[add_vec] loss:%d deg:%f"%(loss,deg*22.5),ap,bp)
    mo = (0,0)
    if len(vec_list)<=2: return (mo, 0, 0)

    f_vec_li = []
    vec_list = sorted(vec_list, key=lambda x:x['loss'])
    vec0_a = vec_list[0]['a'][:2]
    vec0_b = vec_list[0]['b'][:2]
    d_v = (vec0_b[0]-vec0_a[0],vec0_b[1]-vec0_a[1])

    for vec in vec_list:
        if np.linalg.norm(vec['a'][:2]-vec0_a)>3:
            f_vec_li.append(vec)

    vec_deg_li = []; vec_scale_li = []
    for f_vec in f_vec_li:
        vec_a = f_vec['a'][:2]-vec0_a
        vec_b = f_vec['b'][:2]-vec0_b
        vec_a_s, vec_b_s = map(np.linalg.norm,(vec_a,vec_b))
        vec_deg_li.append(np.arccos(np.dot(vec_a,vec_b)/(vec_a_s*vec_b_s+eps)))
        vec_scale_li.append(vec_b_s/vec_a_s)
        print(vec_a,vec_b)
    vec_deg = vec_deg_li[0]
    vec_scale = vec_scale_li[0]
    print(vec_scale,vec_deg)
    return tuple(d_v[0], d_v[1]), vec_scale, vec_deg

# main-code

## load image
temp_img = blank_to_zero(temp_imgs['usagi.ppm'])
back_img = back_imgs['class2_b1_n0_1.ppm']
t_h, t_w = np.shape(temp_img)[:2]
b_h, b_w = np.shape(back_img)[:2]

print('temp_size:(%d,%d)'%(t_h,t_w))
print('image_size:(%d,%d)'%(b_h,b_w))

#plt.plot(temp_hist[0])


a_img = temp_img
b_img = back_img
trim_img, cen_yx, size = hist_mat(a_img, b_img, 0.8)
c_y, c_x = cen_yx[0], cen_yx[1]; s_h, s_w = size[0], size[1]
cv2.imwrite('result/hist_res_img.ppm', trim_img)

move, rescale, redeg = feat_mat(a_img, trim_img, scale=0.8)
print(move, rescale, redeg)
c_y += move[0]
c_x += move[1]
fix_img = b_img[c_y:c_y+s_h, c_x:c_x+s_w]

#rotation & resize
#fix_img = ndimage.rotate(fix_img, np.rad2deg(redeg), reshape=False)
#res_img = cv2.resize(fix_img, None, fx=rescale, fy=rescale)

cv2.imwrite('result/feat_res_img.ppm', fix_img)

print(c_y+int(s_h/2),c_x+int(s_w/2))
