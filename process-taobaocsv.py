from __future__ import absolute_import
import random
from io import open

file = './Taobao_data.csv'
fi = open(file, 'r')
fi.readline()
item_list = []
user_map = {}
fi = open(file, 'r')
fi.readline()
for line in fi:
    item = line.strip().split(',')
    if item[0] not in user_map:
        user_map[item[0]]=[]
    user_map[item[0]].append(("\t".join(item), float(item[-1])))
    item_list.append(item[1])

fi = open(file, 'r')
fi.readline()
meta_map = {} 


for line in fi:
    arr = line.strip().split(",")
    if arr[1] not in meta_map:
        meta_map[arr[1]] = arr[2]


fo = open("jointed-new", "w")
for key in user_map:
    sorted_user_bh = sorted(user_map[key], key=lambda x: x[1])
    for line, t in sorted_user_bh:
        items = line.split("\t")
        asin = items[1]
        j = 0
        items.pop()
        while True:
            asin_neg_index = random.randint(0, len(item_list) - 1)
            asin_neg = item_list[asin_neg_index]
            if asin_neg == asin:
                continue
            items[1] = asin_neg
            print >>fo, "0" + "\t" + "\t".join(items) + "\t" + meta_map[asin_neg]
            j += 1
            if j == 1:
                break
        if asin in meta_map:
            print >>fo, "1" + "\t" + line + "\t" + meta_map[asin]
        else:
            print >>fo, "1" + "\t" + line + "\t" + "default_cat"

fi = open("jointed-new", "r")
fo = open("jointed-new-split-info", "w")
user_count = {}
for line in fi:
    line = line.strip()
    user = line.split("\t")[1]
    if user not in user_count:
        user_count[user] = 0
    user_count[user] += 1
fi.seek(0)
i = 0
last_user = "A26ZDKC53OP6JD"
for line in fi:
    line = line.strip()
    user = line.split("\t")[1]
    if user == last_user:
        if i < user_count[user] - 2:
            print >>fo, "20180118" + "\t" + line
        else:
            print >>fo, "20190119" + "\t" + line
    else:
        last_user = user
        i = 0
        if i < user_count[user] - 2:
            print >>fo, "20180118" + "\t" + line
        else:
            print >>fo, "20190119" + "\t" + line
    i += 1


fin = open("jointed-new-split-info", "r")
ftrain = open("local_train", "w")
ftest = open("local_test", "w")


last_user = "0"
common_fea = ""
line_idx = 0
for line in fin:
    line_idx += 1
    items = line.strip().split("\t")
    ds = items[0] 
    clk = int(items[1])
    user = items[2]
    movie_id = items[3]
    dt = items[6]
    cat1 = items[7]  

    if ds=="20180118":
        fo = ftrain
        tag = 1
    else:
        fo = ftest
        tag = 0
    if user != last_user:
        movie_id_list = []
        cate1_list = []

    else:
        history_clk_num = len(movie_id_list)
        # cat_str = ""
        # mid_str = ""
        # for c1 in cate1_list:
        #     cat_str += c1 + ","
        # for mid in movie_id_list:
        #     mid_str += mid + ","


        if history_clk_num >= 1:    # 8 is the average length of user behavior
            if tag == 1:
                if random.random()<0.2:
                # print >>fo, items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + ','.join(movie_id_list[-100:]) + "\t" + ','.join(cate1_list[-100:])
                    print(items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + ','.join(movie_id_list[-100:]) + "\t" + ','.join(cate1_list[-100:]),file = fo)
            else:
                # print >>fo, items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + ','.join(movie_id_list) + "\t" + ','.join(cate1_list)
                print(items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + ','.join(movie_id_list) + "\t" + ','.join(cate1_list),file = fo)
    last_user = user
    if clk:
        movie_id_list.append(movie_id)
        cate1_list.append(cat1)
    # line_idx += 1
    print(line_idx)
print('local_aggretor finished')
