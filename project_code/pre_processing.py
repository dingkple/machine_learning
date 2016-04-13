# Feature Importance
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.cluster import KMeans
import numpy as np
import json
import sys

DATA_DIR = '/Users/kingkz/Downloads/yelp_dataset_challenge_academic_dataset/'

# VALUE_TYPE = {
#     'city': 'str' ,\
#     'review_count': 'num' ,\
#     'name': 'str' ,\
#     'neighborhoods': 'str' ,\
#     'type': 'str' ,\
#     'business_id': 'other' ,\
#     'full_address': 'str' ,\
#     'hours': 'other' ,\
#     'state': 'str' ,\
#     'longitude': 'num' ,\
#     'stars': 'num' ,\
#     'latitude': 'num' ,\
#     'attributes': 'dict' ,\
#     'open': 'other' ,\
#     'Take-out': 'bool' ,\
#     'Drive-Thru': 'bool' ,\
#     'Outdoor Seating': 'bool' ,\
#     'Caters': 'bool' ,\
#     'Noise Level': 'num' ,\
#     'Parking': 'dict' ,\
#     'Delivery': 'str' ,\
#     'Attire': 'number' ,\
#     'Has TV': 'bool' ,\
#     'Price Range': 'num' ,\
#     'Good For': 'str' ,\
#     'Takes Reservations': 'str' ,\
#     'Waiter Service': 'str' ,\
#     'Accepts Credit Cards': 'str' ,\
#     'Good for Kids': 'str' ,\
#     'Good For Groups': 'str' ,\
#     'Alcohol': 'bool' ,\
#     'romantic': 'bool' ,\
#     'intimate': 'bool' ,\
#     'classy': 'bool' ,\
#     'hipster': 'bool' ,\
#     'divey': 'bool' ,\
#     'touristy': 'bool' ,\
#     'trendy': 'bool' ,\
#     'upscale': 'bool' ,\
#     'casual': 'bool' ,\
# }

main_value_map = {
    'review_count': 'num' ,\
    # 'longitude': 'num' ,\
    # 'latitude': 'num' ,\
}

attribute_map = {
    'Drive-Thru': 'bool', \
    'Good for Kids': 'bool', \
    'Price Range': 'num', \
    'BYOB': 'bool', \
    'Caters': 'bool', \
    'Delivery': 'bool', \

    'Dogs Allowed': 'bool', \
    'Coat Check': 'bool', \
    'Accepts Credit Cards': 'bool', \
    'Take-out': 'bool', \
    'Corkage': 'bool', \
    'By Appointment Only': 'bool', \

    'Happy Hour': 'bool', \
    'Wheelchair Accessible': 'bool', \
    'Outdoor Seating': 'bool', \
    'Takes Reservations': 'bool', \
    'Waiter Service': 'bool', \
    'Good For Dancing': 'bool', \

    'Order at Counter': 'bool', \
    'Has TV': 'bool', \
    'Good For Groups': 'bool', \
}

parking_map = {
    'garage': 'bool', \
    'street': 'bool', \
    'validated': 'bool', \
    'lot': 'bool', \
    'valet': 'bool', \
}

ambience_map = {
    'romantic': 'bool', \
    'intimate': 'bool', \
    'classy': 'bool', \
    'hipster': 'bool', \
    'divey': 'bool', \
    'touristy': 'bool', \
    'trendy': 'bool', \
    'upscale': 'bool', \
    'casual': 'bool', \
}

alcohol_list = [
    'none', \
    'beer_and_wine', \
    'full_bar', \
    'N/A', \
]

noise_list = [
    'very_loud', \
    'average', \
    'loud', \
    'quiet', \
    'N/A', \
]

attire_list = [
    'formal', \
    'dressy', \
    'casual', \
    'N/A', \
]

smoke_list = [
    'yes', \
    'outdoor', \
    'N/A', \
    'no', \
]

wifi_list = [
    'paid', \
    'N/A', \
    'free', \
    'no', \
]

good_for_map = {
    'dessert': 'bool', \
    'latenight': 'bool', \
    'lunch': 'bool', \
    'dinner': 'bool', \
    'brunch': 'bool', \
    'breakfast': 'bool', \
}


name_printed = False
name_count = 0

# \item Accept Insurance: False:2,
# NA: 23402, We Ignore this attribute
# \item Drive-Thru: False: 1798, True: 1475, N/A: 20131, N/A regarded False
# \item Alcohol: none:9391,beer and wine:3090,full bar:7802, N/A:3121
# \item Open 24 Hours: False, True, N/A
# \item Noise Level:  very loud:561, average:12550, loud:1482, quiet:4290, N/A
# \item Music: dj': 1859, background music: 854, jukebox: 1271, live: 1257, video: 1165, karaoke: 866 (Multiple)
# \item Attire: formal:56,dressy:769,casual:21627,N/A:592
# \item Ambience: romantic': 18984, u'intimate': 18984, u'classy': 18984, u'hipster': 18866, u'divey': 18425, u'touristy': 18984, u'trendy': 18984, u'upscale': 18879, u'casual': 18984 (Multiple)
# \item Good for Kids: False,True,N/A
# \item Price Range: 1: 10498, 2: 11240, 3: 1378, 4: 288
# \item BYOB: False: 786, True: 42,N/A: 22576
# \item Caters: False,True,N/A 15276
# \item Delivery: False,True,N/A 21760
# \item Dogs Allowed: False: 1822, True: 543, 'N/A': 21039
# \item Coat Check: False: 2097, True: 158, 'N/A': 21149
# \item Smoking:  'yes': 295, 'outdoor': 1181, 'N/A': 20881, 'no': 1047
# \item Accepts Credit Cards: False: 888, True: 21925, 'N/A': 591
# \item Take-out: False: 2023, True: 20234, 'N/A': 1147
# \item Corkage: False: 490, True: 128, 'N/A': 22786
# \item By Appointment Only: False: 22, True: 2, 'N/A': 23380
# \item Happy Hour: False: 395, True: 1975, 'N/A': 21034
# \item Wheelchair Accessible: False: 1057, True: 10393, 'N/A': 11954
# \item Outdoor Seating: False: 12531, True: 9371, 'N/A': 1502
# \item Takes Reservations: False: 14001, True: 7821, 'N/A': 1582
# \item Waiter Service: False: 7670, True: 12802, 'N/A': 2932
# \item Wi-Fi: 'paid': 155, 'N/A': 6564, u'free': 6396, u'no': 10289
# \item Dietary Restrictions: u'dairy-free': 145, u'gluten-free': 145, u'vegan': 145, u'kosher': 145, u'halal': 145, u'soy-free': 145, u'vegetarian': 145
# \item Good For Dancing: False: 1979, True: 331, 'N/A': 21094
# \item Order at Counter: False: 137, True: 228, 'N/A': 23039
# \item Good For: 'dessert': 21152, u'latenight': 21205, u'lunch': 21205, u'dinner': 21205, u'brunch': 21154, u'breakfast': 21213 (Multiple)
# \item Parking: garage': 20496, u'street': 20494, u'validated': 20306, u'lot': 20494, u'valet': 20494 (Multiple)
# \item Has TV: False: 10105, True: 9920, 'N/A': 3379
# \item Good For Groups: False: 2488, True: 19969, 'N/A': 947


    # # load the iris datasets
    # dataset = datasets.load_iris()
    # # fit an Extra Trees model to the data
    # model = ExtraTreesClassifier()
    # model.fit(dataset.data, dataset.target)
    # # display the relative importance of each attribute
    # print(model.feature_importances_)

data_map_container = {}
location_lst = []
def get_labels():
    model = KMeans(n_clusters=10)
    np.array(location_lst)
    model.fit(location_lst)
    return map(lambda x: 1<<x, model.labels_)

def add_value_map_to_num(dst, data, key_name, value_list):
    global name_printed
    if not name_printed:
        print_key_name(key_name)
    if key_name not in data_map_container:
        data_map_container[key_name] = translate_one_hop(value_list)
    if key_name in data:
        dst.append(data_map_container[key_name][data[key_name]])
    else:
        dst.append(value_list.index('N/A'))

def add_value(dst, data, key_name, value_map):
    # print key_name, data
    global name_printed
    if key_name and key_name in data:
        data = data[key_name]
    elif key_name:
        data = {}
    for k in value_map:
        if not name_printed:
            print_key_name(k)

        if value_map[k] == 'num':
            if k in data:
                dst.append(float(data[k]))
            else:
                dst.append(0)
        elif value_map[k] == 'bool':
            if k in data:
                # print k, data
                if data[k] == 'false':
                    dst.append(0)
                else:
                    dst.append(1)
            else:
                dst.append(2)


def translate_one_hop(a_list):
    key_num = len(a_list)
    rst_map = {}
    for i, key in enumerate(a_list):
        rst_map[key] = 1 << i

    return rst_map

def transform_atrribute(rst, data):
    add_value(rst, data, '', attribute_map)


    add_value_map_to_num(rst, data, 'Wi-Fi', wifi_list)
    add_value_map_to_num(rst, data, 'Smoking', smoke_list)
    add_value_map_to_num(rst, data, 'Attire', attire_list)
    add_value_map_to_num(rst, data, 'Alcohol', alcohol_list)
    add_value_map_to_num(rst, data, 'Noise Level', noise_list)
    add_value(rst, data, 'Parking', parking_map)
    add_value(rst, data, 'Ambience', ambience_map)
    add_value(rst, data, 'Good For', good_for_map)

    global name_printed
    name_printed = True


def print_key_name(k):
    global name_count
    print name_count, k
    name_count += 1

def transform_main_value(data):
    rst = []
    
    for k in main_value_map:
        add_value(rst, data, '', main_value_map)

    transform_atrribute(rst, data['attributes'])
    location_lst.append(np.array([data['longitude'],data['latitude']]))

    return rst


def map_original_data(file_whole_path):

    json_file = open(file_whole_path)

    rst = []
    for line in json_file.readlines():
        rst.append(json.loads(line))

    print len(rst)

    X = []
    Y = []
    
    c = 0

    for d in rst:
        # c += 1
        # if c > 5000:
        #     break
        # print d['hours']
        # print len(d.keys())
        if 'Restaurants' in d['categories'] and 'Price Range' in d['attributes']:
            X.append(transform_main_value(d))
            Y.append(int(d['stars'] * 2))
    labels = get_labels()
    for i,buz in enumerate(X):
        buz.append(labels[i])

    print len(X)

    return np.array(X), np.array(Y)


def feature_select(X, Y):
    # create a base classifier used to evaluate a subset of attributes
    model = LogisticRegression()
    # create the RFE model and select 3 attributes
    rfe = RFE(model, 20)
    rfe = rfe.fit(X, Y)
    # summarize the selection of the attributes
    for i, k in enumerate(rfe.support_):
        print i, k
    for i, k in enumerate(rfe.ranking_):
        print i, k
    # print(rfe.support_)
    # print(rfe.ranking_)


def feature_s_v2(X, Y):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    # display the relative importance of each attribute
    # print(model.feature_importances_)
    for i, k in enumerate(model.feature_importances_):
        print i, k
        

    # for k in rst[0].keys():
    #     print k
    # for a in rst[0]['attributes']:
    #     print a
    # for a in rst[0]['attributes']['Ambience'].keys():
    #     print a
    # for a in rst[0]['attributes']['Good For']:
    #     print a

def print_mean_median(a):
    print 'mean median: ', np.mean(a), np.median(a)


def split_data_set(X, Y):
    train_num = int(len(X) * 0.75)
    train_x, train_y, test_x, test_y = X[:train_num, :], Y[:train_num], X[train_num: , :], Y[train_num:]
    print print_mean_median(test_y)
    return train_x, train_y, test_x, test_y

def svm_test(X, Y):
    train_x, train_y, test_x, test_y = split_data_set(X, Y)

    clf = svm.SVC(C=0.001, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    print 'fitting: '
    clf.fit(train_x, train_y)
    print 'predicting: '
    rst = clf.predict(test_x)
    # print sum(rst == test_y) / len(test_y) * 1.0
    print_mean_median(rst)
    print sum_rst(rst, test_y)

def sum_rst(rst, dst):
    l = 0
    print 'size', len(rst), len(dst)
    for k in range(len(rst)):
        if rst[k] == dst[k]:
            l += 1
    return l * 1.0 / len(rst)

def read_data(dir_path = DATA_DIR):
    file_name = 'yelp_academic_dataset_business.json'
    if dir_path[-1] != '/':
        dir_path += '/'

    fp = dir_path + file_name

    X, Y = map_original_data(fp)
    return X, Y

def main():
    if len(sys.argv) > 1:
        X, Y = read_data(sys.argv[1])
    else:
        X, Y = read_data()
    # svm_test(X, Y)

    # feature_select(X, Y)
    # feature_s_v2(X, Y)

# split_data_set(X, Y)

# a = [[1,2,3], [4,5,6]]

# a = np.array(a)

# print a[:1, :]


if __name__ == '__main__':
    main()


