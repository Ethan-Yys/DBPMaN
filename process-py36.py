import random

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
meta_map = {}  # meta_map记录item_category映射
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
        while True:
            asin_neg_index = random.randint(0, len(item_list) - 1)
            asin_neg = item_list[asin_neg_index]
            if asin_neg == asin:
                continue
            items[1] = asin_neg
            print("0" + "\t" + "\t".join(items) + "\t" + meta_map[asin_neg], file=fo)
            j += 1
            if j == 1:  # negative sampling frequency
                break
        if asin in meta_map:
            print("1" + "\t" + line + "\t" + meta_map[asin], file=fo)
        else:
            print("1" + "\t" + line + "\t" + "default_cat", file=fo)

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
        if i < user_count[user] - 2:  # 1 + negative samples
            print("20180118" + "\t" + line, file=fo)
        else:
            print("20190119" + "\t" + line, file=fo)
    else:
        last_user = user
        i = 0
        if i < user_count[user] - 2:
            print("20180118" + "\t" + line, file=fo)
        else:
            print("20190119" + "\t" + line, file=fo)
    i += 1


fin = open("jointed-new-split-info", "r")
ftrain = open("/opt/local_train", "w")
ftest = open("/opt/local_test", "w")

last_user = "0"
common_fea = ""
line_idx = 0
for line in fin:
    items = line.strip().split("\t")
    ds = items[0] #标记训练/验证集
    clk = int(items[1]) #标记正负样本
    user = items[2]
    movie_id = items[3]
    dt = items[5] #标记时间戳
    cat1 = items[6]  #标记类别（如book）

    if ds=="20180118":
        fo = ftrain
        tag = 1
    else:
        fo = ftest
        tag = 0
    if user != last_user:
        movie_id_list = []
        cate1_list = []
        # print >> fo, items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + "" + "\t" + ""
    else:
        history_clk_num = len(movie_id_list)
        # cat_str = ""
        # mid_str = ""
        # for c1 in cate1_list:
        #     cat_str += c1 + ","
        # for mid in movie_id_list:
        #     mid_str += mid + ","
        # if len(cat_str) > 0: cat_str = cat_str[:-1]
        # if len(mid_str) > 0: mid_str = mid_str[:-1]
        if history_clk_num >= 1:    # 8 is the average length of user behavior
            if tag == 1:
                print(items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + ','.join(movie_id_list[-100:]) + "\t" + ','.join(cate1_list[-100:]), file=fo)
            else:
                print(items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + ','.join(movie_id_list) + "\t" + ','.join(cate1_list), file=fo)
    last_user = user
    if clk:
        movie_id_list.append(movie_id)
        cate1_list.append(cat1)
    line_idx += 1

print('local_aggretor finished')