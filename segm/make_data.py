import os
import glob

data_root = '../../data/DIS5K'
dir_list = ['DIS-TR','DIS-VD']
for dirname in dir_list:
    fid = open(data_root+'/{:}.txt'.format(dirname),'w')
    imglist = glob.glob(data_root+'/'+dirname+'/im/*.jpg')
    for imgpath in imglist:
        imgname = imgpath.split('/')[-1][:-4]
        fid.write('%s\n'%imgname)
    fid.close()