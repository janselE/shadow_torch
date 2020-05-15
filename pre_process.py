import json
import yaml
from PIL import Image, ImageDraw
import os

_sorted_coarse_names = [
        "electronic-things",  # 0
        "appliance-things",  # 1
        "food-things",  # 2
        "furniture-things",  # 3
        "indoor-things",  # 4
        "kitchen-things",  # 5
        "accessory-things",  # 6
        "animal-things",  # 7
        "outdoor-things",  # 8
        "person-things",  # 9
        "sports-things",  # 10
        "vehicle-things",  # 11

        "ceiling-stuff",  # 12
        "floor-stuff",  # 13
        "food-stuff",  # 14
        "furniture-stuff",  # 15
        "rawmaterial-stuff",  # 16
        "textile-stuff",  # 17
        "wall-stuff",  # 18
        "window-stuff",  # 19
        "building-stuff",  # 20
        "ground-stuff",  # 21
        "plant-stuff",  # 22
        "sky-stuff",  # 23
        "solid-stuff",  # 24
        "structural-stuff",  # 25
        "water-stuff"  # 26
        ]

def _find_parent(name, d):
    for k, v in d.items():
        if isinstance(v, list):
              if name in v:
                    yield k
        else:
            assert (isinstance(v, dict))
            for res in _find_parent(name, v):  # if it returns anything to us
                yield res

def generate_fine_to_coarse():
    fine_index_to_coarse_index = {}
    fine_name_to_coarse_name = {}

    with open("/home/jansel/Documents/Research/coco_dataset/data/cocostuff_fine_raw.txt") as f:
        l = [tuple(pair.rstrip().split('\t')) for pair in f]
        l = [(int(ind), name) for ind, name in l]

    with open("/home/jansel/Documents/Research/coco_dataset/data/cocostuff_hierarchy.y") as f:
        d = yaml.load(f, Loader=yaml.FullLoader)

    for fine_ind, fine_name in l:
        assert (fine_ind >= 0 and fine_ind < 182)
        parent_name = list(_find_parent(fine_name, d))
        # print("parent_name of %d %s: %s"% (fine_ind, fine_name, parent_name))
        assert (len(parent_name) == 1)
        parent_name = parent_name[0]
        parent_ind = _sorted_coarse_name_to_coarse_index[parent_name]
        assert (parent_ind >= 0 and parent_ind < 27)

        fine_index_to_coarse_index[fine_ind] = parent_ind
        fine_name_to_coarse_name[fine_name] = parent_name

    assert (len(fine_index_to_coarse_index) == 182)

    return fine_index_to_coarse_index, fine_name_to_coarse_name

def getMaps():
    dic = {}
    index = 0
    seg_list = []
    cat_list = []
    curr = 0
    len_img = len(data['images'])
    for index in range(len_img):
        img_id = data['images'][index]
        path = img_id['file_name']
        len_ann = len(data['annotations'])
        ann_ids = []
        bad = False

        for i in range(len_ann):
            if data['annotations'][i]['image_id'] == data['images'][index]['id']:
                ann_ids.append(i)

        num_objs = len(ann_ids)
        seg_list = []
        cat_list = []

        for i in range(num_objs):
            seg = data['annotations'][ann_ids[i]]['segmentation']
            cat = data['annotations'][ann_ids[i]]['category_id']
            crowd = data['annotations'][ann_ids[i]]['iscrowd']

            seg_list.append(seg)
            cat_list.append(fine_index_to_coarse_index[cat])

            if crowd == 1:
                bad = True
                break
            for n in range(len(seg)):
                lab = fine_index_to_coarse_index[cat]
                class_name = _sorted_coarse_names[lab]

                if lab >= 12 or bad:
                    bad = True
                    break
        if bad:
            continue
        dic[str(curr)] = [path, seg_list, cat_list]
        curr += 1

        if index % 100 == 0:
            print("index {} of {}".format(index, len_img))
    

    return dic


annotation = '/home/jansel/Documents/Research/coco_dataset/data/instances_train2017.json'
root = '/home/jansel/Documents/Research/coco_dataset/data/'

with open(annotation) as f:
    data = json.load(f)

print('done loading data')

_sorted_coarse_name_to_coarse_index = {n: i for i, n in enumerate(_sorted_coarse_names)}
fine_index_to_coarse_index, fine_name_to_coarse_name = generate_fine_to_coarse()

print('done generating fine to coarse')


import timeit

start = timeit.default_timer()

dic = getMaps()


stop = timeit.default_timer()

print('Time: ', stop - start)  


print('starting mask creation')
len_dic = len(dic)
for index in range(len_dic):
    path = dic[str(index)][0]
    
    img = Image.open(os.path.join(root + 'train2017', path))
    img = img.convert('RGB')
    num_objs = len(dic[str(index)][1])
    w, h = img.size
    
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    
    for i in range(num_objs):
        seg = dic[str(index)][1][i]
        cat = dic[str(index)][2][i]
    
        for n in range(len(seg)):
            draw.polygon(seg[n], outline=None, fill=cat)

    mask.save(os.path.join(root + 'train2017_mask', path))
print('done creating mask')
