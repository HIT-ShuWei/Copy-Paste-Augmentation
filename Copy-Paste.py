from numpy.lib.twodim_base import vander
from pycocotools.coco import COCO
from tqdm import tqdm
import argparse
import os

import numpy as np
import cv2 as cv
import json

def random_select_img(imgIds):
    # 根据imgIds随机选取两张图片，第一、二张依次为src_img,main_img
    (src_img, main_img) = np.random.choice(imgIds, size=2, replace=False, p=None)
    return (int(src_img), int(main_img))

def random_select_anns(annIds):
    # 随机选取当前src_img的anns
    total_num = len(annIds)     # anns的总数目
    if total_num == 1:
        return annIds
    # print('total_num',total_num)
    select_num = np.random.randint(1,total_num)   # 选取anns的数目
    select_annIds = np.random.choice(annIds, size=select_num, replace=False, p=None)
    return list(select_annIds)

    
def random_flip_horizontal(img, mask_imgs, p=0.5):
    '''
    随机水平翻转，并处理bbox和seg标签，翻转概率默认为0.5
    img: 原始的完整图片
    mask_img: 当前图片中anns截取下来的图片的List
    '''
    if np.random.random() < p:
        # 概率筛选
        img = img[:, ::-1, :]
        new_mask_imgs = []
        for i, mask_img in enumerate(mask_imgs):
            mask_img = mask_img[:, ::-1, :]
            new_mask_imgs.append(mask_img)
        return img, new_mask_imgs
    else:
        return img, mask_imgs

def LSJ(img, mask_imgs, min_scale=0.1, max_scale=2.0):
    '''
    Large-Scaling-Jittering
    img: 原始的完整图片
    mask_img: 当前图片中anns截取下来的图片的List
    '''
    # 得到rescale比例
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = img.shape
    # 得到新尺寸
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv.resize(img, (w_new, h_new), interpolation=cv.INTER_LINEAR)
    new_mask_imgs = []
    for i, mask_img in enumerate(mask_imgs):
        mask_img = cv.resize(mask_img, (w_new, h_new), interpolation=cv.INTER_NEAREST)
        new_mask_imgs.append(mask_img)
    mask_imgs = new_mask_imgs
    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:
        # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        img_pad[y:y+h_new, x:x+w_new, :] = img
        mask_imgs_pad = []
        for i, mask_img in enumerate(mask_imgs):
            mask_img_pad = np.zeros((h, w, 3), dtype=np.uint8)
            mask_img_pad[y:y+h_new, x:x+w_new] = mask_img
            mask_imgs_pad.append(mask_img_pad)
        return img_pad, mask_imgs_pad
    else:
        # crop
        img_crop = img[y:y+h, x:x+w, :]
        mask_imgs_crop = []
        for i, mask_img in enumerate(mask_imgs):
            mask_img_crop = mask_img[y:y+h, x:x+w]
            mask_imgs_crop.append(mask_img_crop)
        return img_crop, mask_imgs_crop

def get_annos_from_mask(mask_imgs):
    '''
    利用mask得到新的annotations，需要考虑最大尺寸，anno不能超过最大尺寸
    '''
    new_anns = []
    for i, mask_img in enumerate(mask_imgs):
        mask_img = cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY)
        y, x = mask_img.nonzero()   # 找到图像中非0像素位置
        if y.size == 0 or x.size == 0:
            # 说明当前目标被LSJ放缩之后太小了，已经没有mask了，跳过
            bbox = []
            new_anns.append(bbox)
            continue
        ymax, ymin, xmax, xmin = max(y), min(y), max(x), min(x) # 找到坐标框边界值
        w, h = xmax-xmin, ymax-ymin
        bbox = [xmin, ymin, w, h]
        new_anns.append(bbox)
    return new_anns


def add_mask_to_img(main_img, mask_imgs):
    '''
    将新的mask添加当img上
    输入：
    1.main_img: 在此图片上添加mask图片
    2.mask_imgs: list,其中每个元素为背景为黑色(0)，只截取了目标部分的图像
    '''    
    # 获取main_img尺寸
    if len(main_img.shape) == 3:
        h,w,c = main_img.shape
    elif len(main_img.shape) == 2:
        h, w = main_img.shape
    #---------------------------------------------#
    #   扣掉main_img对应位置的图像，然后贴上mask的图片
    #---------------------------------------------#
    new_mask_imgs = []
    for i, mask_img in enumerate(mask_imgs):

        # 对mask预处理，确保其不超出main的尺寸
        mask_img = cv.resize(mask_img, (w, h), interpolation=cv.INTER_NEAREST)
        
        mask = cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY)                      # 获得单通道灰度mask_img
        ret, mask = cv.threshold(mask, 0.0000000001, 255, cv.THRESH_BINARY)  # 二值化处理
        mask_inv = cv.bitwise_not(mask)                                      # 非运算，mask取反，用于扣掉图片
        main_img = cv.bitwise_and(main_img, main_img, mask=mask_inv)         # 删除了img中的mask_inv区域
        main_img = cv.add(main_img, mask_img)                                # 贴图
        new_mask_imgs.append(mask_img)

    return main_img, new_mask_imgs

def gen_new_anns(imgs, file_names, labels, origin_anns, cats):
    '''
    1.生成新的anns,将新bbox标签替代原始anns中的bbox标签
    2.清洗掉没有Bbox的anns
    '''
    # json文件中包括3个字典，images,annotations,categories
    images, annotations, categories = [],[], cats
    ann_id = 10062
    for i, origin_ann in enumerate(origin_anns):
        images.append({
            'file_name':file_names[i],
            'height':imgs[i].shape[0],
            'width':imgs[i].shape[1],
            'id':i+1719
        })
        for j, label in enumerate(labels[i]):
            if len(label) == 0:
                # 去掉空的bbox
                continue
            annotations.append({
                'bbox': [int(l) for l in label],
                'area': int(label[2]*label[3]),
                'iscrowd':0,
                'category_id': origin_ann[j]['category_id'],
                'image_id':i+1719,
                'id':ann_id
            })
            ann_id += 1
    return images, annotations, categories



def copy_paste(main_img, main_mask_imgs, src_img, src_mask_imgs):
    '''
    Copy-Paste的主体实现,主要流程如下：
    1.random_flip
    2.Large Scale Jittering
    3.src上的anns融合到main上，更新anns
    
    输入：1.main_img, src_img ： 图片
         2.main_mask_imgs, src_mask_imgs：list，存储各个依靠seg截取下来的目标的图像，各自都是独立的
    输出：copy_paste_img, copy_paste_anns
    '''
    # random flip
    main_img, main_mask_imgs = random_flip_horizontal(main_img, main_mask_imgs)
    src_img, src_mask_imgs = random_flip_horizontal(src_img, src_mask_imgs)

    # LSJ
    main_img, main_mask_imgs =LSJ(main_img, main_mask_imgs)
    src_img, src_mask_imgs = LSJ(src_img, src_mask_imgs)
    
    # 将src_mask粘贴到main_img上
    main_img, new_src_mask_imgs = add_mask_to_img(main_img, src_mask_imgs)

    # 根据mask更新annotations
    new_anns_1 = get_annos_from_mask(main_mask_imgs)       # 原图片的mask
    new_anns_2 = get_annos_from_mask(new_src_mask_imgs)    # 新贴上的mask
    new_anns = new_anns_1 + new_anns_2  #anns合并
    copy_paste_img, copy_paste_anns = main_img, new_anns

    return copy_paste_img, copy_paste_anns


def main(args):
    '''
    Copy-Paste实现流程：
    1.读取数据
    2.随机选取src/main img以及src的annotation
    3.Copy-Paste
    4.后续处理图片、生成Json文件
    '''
    #--------------------------------------#
    #               1.读取数据
    #--------------------------------------#
    # anno_file = os.path.join(args.input_dir, 'annotations', 'instances_train2017.json')  # annotations路径
    anno_file = os.path.join(args.input_dir, 'annotations', 'train.json')  # annotations路径
    img_file = os.path.join(args.input_dir, 'train2017')                   # .png / .jpg路径
    coco = COCO(anno_file)      # 读取标签
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    cats = coco.loadCats(catIds)
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))

    # 创建输出路径
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'annotations'), exist_ok=True)    # 标签存储文件
    os.makedirs(os.path.join(args.output_dir, 'train2017'), exist_ok=True)      # 图片存储文件

    target_nums = args.aug_ratio * len(imgIds)  # 产生增广图片的目标数量
    # target_nums = 10   # 测试用
    flag = 0
    
    new_img_filenames = []
    new_labels = []
    new_imgs = []
    origin_anns = []
    while flag in tqdm(range(target_nums)):
        flag += 1
        #--------------------------------------#
        #               2.随机选取
        #--------------------------------------#
        # 随机选取img，并且用while循环避免空图片(no anns)
        (src_imgId, main_imgId) = random_select_img(imgIds)
    

        # 读取main的anns
        main_img_info = coco.loadImgs(main_imgId)[0]    # 当前图片信息
        main_img = cv.imread(os.path.join(img_file, main_img_info['file_name']))  # 读取main的图片
        main_annIds = coco.getAnnIds(imgIds=main_img_info['id'], catIds=catIds, iscrowd=None)    #当前图片annIDs的汇总
        if len(main_annIds) == 0:
            # 如果当前图片没有标签，则跳过
            flag -= 1
            continue
        main_anns = coco.loadAnns(main_annIds)       #当前图片anns汇总

        # 获取main中全部的目标截取下来
        main_mask_imgs = []
        for i, main_ann in enumerate(main_anns):
            if isinstance(main_ann['segmentation'][0], int):
                #segmentation应该是一个list嵌套进list的结构，如果只有一个list,说明只有一个实例，且少一层嵌套
                #针对ALET的操作，其他数据集可以注释掉
                main_ann['segmentation'] = [main_ann['segmentation']]

            mask = coco.annToMask(main_ann)  # 二维矩阵，尺寸等于当前图片的尺寸
            mask = np.expand_dims(mask,axis=2).repeat(3,axis=2) #转换为3通道
            mask_img = main_img * mask       # 当前图片相乘，截取seg下来
            main_mask_imgs.append(mask_img)

        # 读取并随机选取src的anns
        src_img_info = coco.loadImgs(src_imgId)[0]
        src_img = cv.imread(os.path.join(img_file, src_img_info['file_name']))
        src_annIds = coco.getAnnIds(imgIds=src_img_info['id'], catIds=catIds, iscrowd=None)
        if len(src_annIds) == 0:
            # 如果当前图片没有标签，则跳过
            flag -= 1
            continue
        src_annIds = random_select_anns(src_annIds)     # 随机选取anns
        src_anns = coco.loadAnns(src_annIds)

        # 获取src中随机选取得到的目标截取下来
        src_mask_imgs = []
        for i, src_ann in enumerate(src_anns):
            if isinstance(src_ann['segmentation'][0], int):
                #segmentation应该是一个list嵌套进list的结构，如果只有一个list,说明只有一个实例，且少一层嵌套
                #针对ALET的操作，其他数据集可以注释掉
                src_ann['segmentation'] = [src_ann['segmentation']] 
            mask = coco.annToMask(src_ann)  # 二维矩阵，尺寸等于当前图片的尺寸
            mask = np.expand_dims(mask,axis=2).repeat(3,axis=2) #转换为3通道
            mask_img = src_img * mask       # 当前图片相乘，截取seg下来
            src_mask_imgs.append(mask_img)
        
        #--------------------------------------#
        #               3.Copy-Paste
        #--------------------------------------#
        copy_paste_img, copy_paste_label = copy_paste(main_img, main_mask_imgs, src_img, src_mask_imgs)


        #--------------------------------------#
        #               4.后续处理
        #--------------------------------------#
        img_filename = 'copy_paste'+str(flag)+'.jpg'
        new_img_filenames.append(img_filename)
        new_imgs.append(copy_paste_img)
        new_labels.append(copy_paste_label)
        origin_anns.append(main_anns+src_anns)
        cv.imwrite(os.path.join(args.output_dir, 'train2017', img_filename), copy_paste_img)
        # cv.imwrite(os.path.join(args.output_dir, 'train2017', 'main_img'+str(flag)+'.jpg'), main_img)
        # cv.imwrite(os.path.join(args.output_dir, 'train2017', 'src_img'+str(flag)+'.jpg'), src_img)
        # cv.imwrite('cp'+str(flag)+'.jpg', copy_paste_img)
        # cv.imwrite('main'+str(flag)+'.jpg', main_img)
        # cv.imwrite('src'+str(flag)+'.jpg', src_img)
    
    # 将ann存储为新的json文件
    images, annotations, categories = gen_new_anns(new_imgs, new_img_filenames, new_labels, origin_anns ,cats)  # 生成新anns
    new_dict = {"images":images, "annotations":annotations, "categories":categories}
    # print('new_json',new_dict)
    # print('annotations:',type(annotations[0]['bbox'][1]))
    new_json = json.dumps(new_dict)
    f1 = open(os.path.join(args.output_dir+'annotations/copy_paste_train.json'), 'w')
    f1.write(new_json)
    f1.close()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/data/Datasets/coco/ALET", type=str,
                        help="root path for coco-like datasets")
    parser.add_argument("--output_dir", default="/data/Datasets/coco/ALET/Copy-Paste", type=str,
                        help="output dataset directory")
    parser.add_argument("--aug_ratio", default=4, type=int, help="aug ratio to origin dataset")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
