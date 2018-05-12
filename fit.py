import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy import signal,ndimage
import glob

RANK = 100

# file read
temp_files = glob.glob('template/*.ppm')
img_files = glob.glob('ppm/*.ppm')

t_imgs = {t.split('/')[-1]:cv2.imread(t) for t in temp_files}
c_imgs = {c.split('/')[-1]:cv2.imread(c) for c in img_files}

# filter
fil_a = np.array([[1.]*3,[0.]*3,[-1.]*3])
fil_b = fil_a.transpose()
fil = fil_a + fil_b
print(fil)
#fil_ = lambda d: ndimage.rotate(fil_0, d, reshape=False)

# lambda fanc
norm = lambda v: v/np.sum(v)
argmax_2d = lambda x,i: list(zip(*np.unravel_index(np.argsort(x.ravel())[::-1],x.shape)))[:i]

#def fanc
def blank_to_zero(img):
    img[np.all(img==255, axis=2)]=0
    return img


def hist_rgb(img):
    rgb = cv2.split(img)
    hists = (np.histogram(x.ravel(),255,[1,255])[0] for x in rgb)
    return np.dstack(hists)

def conv_rgb(img):
    rgb = cv2.split(img)
    res = (signal.convolve2d(x, fil, 'same') for x in rgb)
    return np.dstack(res)

def get_fp(img,r):
    img = conv_rgb(img)
    rgb = cv2.split(img)
    return [argmax_2d(i,r) for i in rgb]

def rotate_match(img,temp_feat):
    img_feat = conv_rgb(img)
    fanc = lambda r: np.linalg.norm(img_feat-np.roll(temp_feat,r))
    loss_list = [fanc(r) for r in range(R_NUM)]
    arg = np.argmin(loss_list)
    return loss_list, arg

def fp_writer(arg_list,shape, name='feature_point'):
    r, g, b = cv2.split(np.zeros(shape))
    for i in range(RANK):
        r[arg_list[0][i]] = 255
        g[arg_list[1][i]] = 255
        b[arg_list[2][i]] = 255
    point_img = np.dstack((r,g,b))
    cv2.imwrite('result/'+name+'.ppm',point_img)
    return point_img

def make_feat_data(f_map, points):
    rad = 3; div=16
    coords = []
    for i in range(div):
        r = 2*np.pi/div
        y = int(np.round(rad * np.sin(r*i)))
        x = int(np.round(rad * np.cos(r*i)))
        coords.append((y,x))
        #print(y,x)

    d0, d1, d2 = f_map.shape
    point_val = np.zeros([d0, d1, d2, div])
    for (i,point) in enumerate(points):
        for p in point:
            py,px = p
            if py<rad or px<rad or py+rad>=d0 or px+rad>=d1: continue
            point_val[py,px,i] = [f_map[py+c[0],px+c[1],i] for c in coords]
            #print(p,point_val[py,px,i])
    return point_val


# load img
temp_img = blank_to_zero(t_imgs['inu.ppm'])
c_img = c_imgs['class1_b1_n0_1.ppm']
t_h, t_w = np.shape(temp_img)[:2]
c_h, c_w = np.shape(c_img)[:2]

print('temp_size:(%d,%d)'%(t_h,t_w))
print('image_size:(%d,%d)'%(c_h,c_w))

temp_hist = hist_rgb(temp_img)
plt.plot(temp_hist[0])


# histgram matching

# init
min_s = 0
min_x, min_y = (0, 0)
re_x, re_y = (0, 0)
stride = 10

# fit
for y in range(0,c_h-t_h,stride):
    for x in range(0,c_w-t_w,stride):
        trim = c_img[y:y+t_h,x:x+t_w]
        c_hist = hist_rgb(trim)
        score = np.sum(np.minimum(temp_hist,c_hist))
        if min_s < score:
            min_s = score
            min_x, min_y = x, y
        elif min_s == score:
            re_x, re_y = x, y
cen_x, cen_y = (int((re_x+min_x)/2), int((re_y+min_y)/2))
print(cen_y,cen_x)



#feature matching
arg_list = get_fp(temp_img,RANK)
p_img = fp_writer(arg_list, temp_img.shape, name='temp_feature')
val_img = make_feat_data(temp_img, arg_list)
cv2.imwrite('result/val_img.ppm', np.sum(val_img, axis=3))

extra_img = c_img[cen_y:cen_y+t_h,cen_x:cen_x+t_w]
feat_p = get_fp(extra_img, RANK)
b_val_img = make_feat_data(extra_img, feat_p)
cv2.imwrite('result/b_val_img.ppm', np.sum(b_val_img, axis=3))
fp_writer(feat_p, extra_img.shape, name='back_feature')


cv2.imwrite('result/res_img.ppm', extra_img)
