# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import os

def get_dim(db_name):

    if db_name=='LDL_data':
        return 30, 150, 5, 32, 9

# return sub_num, clip_num, channels, time-steps, fea_dim
def get_num(db_name):

    if db_name=='LDL_data':
        return 194, 9, 30, 150, 5

def data_preprocess(data, db_name):
    sub_num, clip_num, channels, fea_dim = get_num(db_name)
    num = data.shape[0]
    reshape = np.reshape(data, [num, -1])
    mean = np.mean(reshape, axis=1)
    mean = np.reshape(mean, [-1,1])
    std = np.std(reshape, axis=1)
    std = np.reshape(std, [-1,1])
    norm = (reshape - mean) / std
    data = norm

    return data

#20200724
def get_data_info(path):
    # get data
    path = path #"EEG_PSD_multilabel_9_addLabel_sum1/"  # EEG_PSD_multilabel_4&single_label_sum1/"  # 文件夹目录
    files = sorted(os.listdir(path))  # 得到文件夹下的所有文件名称
    c = 0
    all_sub = []

    for s in range(11, 205):
        sin_sub = []
        clip = 0
        for file in files:  # 遍历文件夹
            c = c + 1
            if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                # get the sub_id & clip_id
                if file[0:3] == 'sub':  # 判断是否为数据文件
                    if file[5:6] == '_':
                        sub_id = file[3:5]
                        clip_id = file[6:7]
                    else:
                        sub_id = file[3:6]
                        clip_id = file[7:8]

                    if int(sub_id) == s:
                        clip = clip + 1
                        if clip <= 9:
                            data_set = sio.loadmat(path + file)  # 读取数据文件中的数据
                            sin_sub.append(data_set)

        all_sub.append(sin_sub)

    # get subject info
    data = all_sub
    s_len = len(data)
    info = []
    for i in range(s_len):
        sub = []
        sub.append(i)
        c_len = len(data[i])
        sub.append(c_len)
        l = []
        for c in range(c_len):
            feature = data[i][c]['feature_arr']
            l.append(feature.shape[0])
        sub.append(l)  # 每一个sub的一个clip的长度
        info.append(sub)


    return all_sub, info

def get_sample_data(path1,path2):

    data,info  = get_data_info(path1)###########
    win_size = 30 #choices=[10,20,30]
    strides = 10 #choices=[1,10,20]

    n_feature = sio.loadmat(path2)['feature_arr'] #'EEG_PSD_multilabel_9_win/featureAll.mat

    f = 0
    all_sub = []
    for n_s in range(len(info)):
        sin_sub = []
        for n_len in range(info[n_s][1]):
            f_num = info[n_s][2][n_len]
            feature = n_feature[f:f + f_num, :]
            data[n_s][n_len]['feature_arr'] = feature
            f = f + f_num

            # split samples by window_size
            t_len = data[n_s][n_len]['feature_arr'].shape[0]

            sub_sample = []
            sub_label = []
            sub_dis = []
            sub_dom_label = []
            sub_score = []

            s_len = t_len
            start = 0
            end = 0
            count = 0
            while end <= s_len:
                count = count + 1
                end = start + win_size
                if end > s_len:
                    break
                print('sample nums....', count)
                sample_feature = data[n_s][n_len]['feature_arr'][start:end, :]
                sample_label = data[n_s][n_len]['label_arr'][start:start + 1, :]  ##############
                sample_dis = data[n_s][n_len]['dis_arr'][start:start + 1, :]
                sample_dom_label = data[n_s][n_len]['primary_arr'][start:start + 1, :]
                sample_score = data[n_s][n_len]['score_arr'][start:start + 1, :]
                start = start + strides

                sub_sample.append(sample_feature)
                sub_label.append(sample_label)
                sub_dis.append(sample_dis)
                sub_dom_label.append(sample_dom_label)
                sub_score.append(sample_score)

            sub_sample = np.asarray(sub_sample)
            sub_label = np.reshape(np.asarray(sub_label), [np.asarray(sub_label).shape[0],
                                                           np.asarray(sub_label).shape[1] * np.asarray(sub_label).shape[
                                                               2]])
            sub_dis = np.asarray(sub_dis)
            sub_dis = np.reshape(np.asarray(sub_dis), [np.asarray(sub_dis).shape[0],
                                                       np.asarray(sub_dis).shape[1] * np.asarray(sub_dis).shape[2]])
            sub_dom_label = np.asarray(sub_dom_label)
            sub_dom_label = np.reshape(np.asarray(sub_dom_label), [np.asarray(sub_dom_label).shape[0],
                                                       np.asarray(sub_dom_label).shape[1] * np.asarray(sub_dom_label).shape[2]])
            sub_score = np.asarray(sub_score)
            sub_score = np.reshape(np.asarray(sub_score), [np.asarray(sub_score).shape[0],
                                                           np.asarray(sub_score).shape[1] * np.asarray(sub_score).shape[
                                                               2]])

            data[n_s][n_len]['feature_arr'] = sub_sample
            data[n_s][n_len]['label_arr'] = sub_label
            data[n_s][n_len]['dis_arr'] = sub_dis
            data[n_s][n_len]['primary_arr'] = sub_dom_label
            data[n_s][n_len]['score_arr'] = sub_score

            sin_sub.append(data[n_s][n_len])

        all_sub.append(sin_sub)
        print('all sub....', all_sub[0][0]['feature_arr'].shape)
        print('all sub....', all_sub[0][1]['feature_arr'].shape)
        print('all sub....', all_sub[0][2]['feature_arr'].shape)

    return all_sub

#20210406
def get_sample_data_withoutOverlap(path1,path2):

    data,info  = get_data_info(path1)###########
    win_size = 150 #30 #choices=[10  20  30]
    strides = win_size

    n_feature = sio.loadmat(path2)['feature_arr'] #'EEG_PSD_multilabel_9_win/featureAll.mat

    f = 0
    all_sub = []
    for n_s in range(len(info)):
        sin_sub = []
        for n_len in range(info[n_s][1]):
            f_num = info[n_s][2][n_len]
            feature = n_feature[f:f + f_num, :]
            data[n_s][n_len]['feature_arr'] = feature
            f = f + f_num

            # split samples by window_size
            t_len = data[n_s][n_len]['feature_arr'].shape[0]

            sub_sample = []
            sub_label = []
            sub_dis = []
            sub_dom_label = []
            sub_score = []

            s_len = t_len
            start = 0
            end = 0
            count = 0
            while end <= s_len:
                count = count + 1
                end = start + win_size
                if end > s_len:
                    break
                print('sample nums....', count)
                sample_feature = data[n_s][n_len]['feature_arr'][start:end, :]
                sample_label = data[n_s][n_len]['label_arr'][start:start + 1, :]  ##############
                sample_dis = data[n_s][n_len]['dis_arr'][start:start + 1, :]
                sample_dom_label = data[n_s][n_len]['primary_arr'][start:start + 1, :]
                sample_score = data[n_s][n_len]['score_arr'][start:start + 1, :]
                start = start + strides

                sub_sample.append(sample_feature)
                sub_label.append(sample_label)
                sub_dis.append(sample_dis)
                sub_dom_label.append(sample_dom_label)
                sub_score.append(sample_score)

            print('sub_sample len....', len(sub_sample))
            sub_sample = np.asarray(sub_sample)
            sub_label = np.reshape(np.asarray(sub_label), [np.asarray(sub_label).shape[0],np.asarray(sub_label).shape[1] * np.asarray(sub_label).shape[2]])
            sub_dis = np.asarray(sub_dis)
            sub_dis = np.reshape(np.asarray(sub_dis), [np.asarray(sub_dis).shape[0],np.asarray(sub_dis).shape[1] * np.asarray(sub_dis).shape[2]])
            sub_dom_label = np.asarray(sub_dom_label)
            sub_dom_label = np.reshape(np.asarray(sub_dom_label), [np.asarray(sub_dom_label).shape[0],np.asarray(sub_dom_label).shape[1] * np.asarray(sub_dom_label).shape[2]])
            sub_score = np.asarray(sub_score)
            sub_score = np.reshape(np.asarray(sub_score), [np.asarray(sub_score).shape[0],np.asarray(sub_score).shape[1] * np.asarray(sub_score).shape[2]])

            data[n_s][n_len]['feature_arr'] = sub_sample
            data[n_s][n_len]['label_arr'] = sub_label
            data[n_s][n_len]['dis_arr'] = sub_dis
            data[n_s][n_len]['primary_arr'] = sub_dom_label
            data[n_s][n_len]['score_arr'] = sub_score

            sin_sub.append(data[n_s][n_len])

        all_sub.append(sin_sub)
        print('all sub....', all_sub[0][0]['feature_arr'].shape)
        print('all sub....', all_sub[0][1]['feature_arr'].shape)
        print('all sub....', all_sub[0][2]['feature_arr'].shape)

    return all_sub



#20210320
def dataSplit(path1,all_sub,db_name ):
    sub_num, clip_num, channels, time_steps, fea_dim = get_num(db_name)
    data = all_sub
    print('all_sub len:',len(data))
    # # train:validation:test = 5:3:2
    fold = 10
    sub = len(data) #194
    print('sub num.....', sub)
    fold_len = sub // fold
    print('fold_len....', fold_len)
    #train:[0:94],val:[95:151],test:[152:193]
    train_s = 0 * fold_len
    train_e = 5 * fold_len
    val_s = 5 * fold_len + 1
    val_e = (5+3) * fold_len
    test_s = (5+3) * fold_len + 1
    test_e = sub - 1

    train_set = data[train_s:train_e]
    val_set = data[val_s:val_e]
    test_set = data[test_s:test_e]

    #train data
    for s in range(len(train_set)):

        print('sub.....', s)
        for v in range(len(train_set[s])):
            print('video clip....', v)
            # print(train_set[s][v]['feature_arr'].shape)
            if s == 0:
                feature = train_set[s][v]['feature_arr']
                label = train_set[s][v]['label_arr']
                dis = train_set[s][v]['dis_arr']
                dom_label = train_set[s][v]['primary_arr']
                score = train_set[s][v]['dis_arr']
            else:
                feature = np.vstack((feature,train_set[s][v]['feature_arr']))
                label = np.vstack((label,train_set[s][v]['label_arr']))
                dis = np.vstack((dis,train_set[s][v]['dis_arr']))
                dom_label = np.vstack((dom_label, train_set[s][v]['primary_arr']))
                score = np.vstack((score,train_set[s][v]['dis_arr']))
    train_data = feature
    train_label = label
    train_dis = dis
    train_dom_label = dom_label
    train_score = score
    print('train_data shape:',train_data.shape) #(162986, 10, 150)
    # print('train_label shape:',train_label.shape) #(162986, 9)
    print('train_dis shape:',train_dis.shape) #(162986, 9)
    # print('train_score shape:',train_score.shape) #(162986, 9)
    print('train_dom_label shape:', train_dom_label.shape)  # (162986, 9)

    #val data
    for s in range(len(val_set)):

        print('sub.....', s)
        for v in range(len(val_set[s])):
            print('video clip....', v)
            # print(val_set[s][v]['feature_arr'].shape)
            if s == 0:
                feature = val_set[s][v]['feature_arr']
                label = val_set[s][v]['label_arr']
                dis = val_set[s][v]['dis_arr']
                dom_label = val_set[s][v]['primary_arr']
                score = val_set[s][v]['dis_arr']
            else:
                feature = np.vstack((feature,val_set[s][v]['feature_arr']))
                label = np.vstack((label,val_set[s][v]['label_arr']))
                dis = np.vstack((dis,val_set[s][v]['dis_arr']))
                dom_label = np.vstack((dom_label, val_set[s][v]['primary_arr']))
                score = np.vstack((score,val_set[s][v]['dis_arr']))
    val_data = feature
    val_label = label
    val_dis = dis
    val_score = score
    val_dom_label = dom_label
    print('val_data shape:',val_data.shape) #(101486, 10, 150)
    # print('val_label shape:',val_label.shape) #(101486, 9)
    print('val_dis shape:',val_dis.shape) #(101486, 9)
    # print('val_score shape:',val_score.shape) #(101486, 9)
    print('val_dom_label shape:', val_dom_label.shape)  # (101486, 9)

    #test data
    for s in range(len(test_set)):

        print('sub.....', s)
        for v in range(len(test_set[s])):
            print('video clip....', v)
            # print(test_set[s][v]['feature_arr'].shape)
            if s == 0:
                feature = test_set[s][v]['feature_arr']
                label = test_set[s][v]['label_arr']
                dis = test_set[s][v]['dis_arr']
                dom_label = test_set[s][v]['primary_arr']
                score = test_set[s][v]['dis_arr']
            else:
                feature = np.vstack((feature,test_set[s][v]['feature_arr']))
                label = np.vstack((label,test_set[s][v]['label_arr']))
                dis = np.vstack((dis,test_set[s][v]['dis_arr']))
                dom_label = np.vstack((dom_label, test_set[s][v]['primary_arr']))
                score = np.vstack((score,test_set[s][v]['dis_arr']))
    test_data = feature
    test_label = label
    test_dis = dis
    test_score = score
    test_dom_label = dom_label
    print('test_data shape:',test_data.shape) #(68832, 10, 150)
    # print('test_label shape:',test_label.shape) #(68832, 9)
    print('test_dis shape:',test_dis.shape) #(68832, 9)
    # print('test_score shape:',test_score.shape) #(68832, 9)
    print('test_dom_label shape:', test_dom_label.shape)  # (68832, 9)

    return train_data, val_data,test_data, train_dis, val_dis, test_dis,train_dom_label, val_dom_label, test_dom_label

#20210406
#20210406
def five_fold(all_sub, fold_id, db_name):
    sub_num, clip_num, channels, time_steps, fea_dim = get_num(db_name)
    data = all_sub
    # five fold cross-valid
    fold = 5
    fold_id = fold_id
    # print('fold_id.....', fold_id)
    sub = len(data)
    # print('sub num.....', sub)
    fold_len = sub // fold
    # print('fold_len....', fold_len)
    fold_s = (fold_id - 1) * fold_len
    if fold_id == 5:
        fold_e = fold_id * fold_len + 4
    else:
        fold_e = fold_id * fold_len

    t_data = []
    tt_data = []
    train_v_len = 0
    test_v_len = 0

    for s in range(sub):
        count = 0
        tag = 0
        # print('s.....', s)
        for ss in range(fold_s,fold_e):
            # print('fold start....',fold_s)
            # print('fold end.....', fold_e)
            count = count + 1
            # print('count.....', count)
            if s == ss:
                tag = 1
                # print('-------')
                test_clip = len(data[s])
                if test_clip != 0:
                    tt_data.append(data[ss])
                    test_v_num = len(data[ss])
                    for test_v_n in range(test_v_num):
                        test_v_len = test_v_len + data[ss][test_v_n]['feature_arr'].shape[0]
            else:
                if count == fold_e - fold_s and tag == 0:
                    # print('*********')
                    # print('fold_e - fold_s = count..', count)
                    t_clip = len(data[s])
                    if t_clip != 0:
                        t_data.append(data[s])
                        train_v_num = len(data[s])
                        for t_v_n in range(train_v_num):
                            train_v_len = train_v_len + data[s][t_v_n]['feature_arr'].shape[0]


    test_data = np.empty((test_v_len, time_steps, 150))
    test_label = np.empty((test_v_len, 9))
    test_dis = np.empty((test_v_len, 9))
    test_dom = np.empty((test_v_len, 9))
    test_score = np.empty((test_v_len, 9))

    tt_sub = len(tt_data)

    _start = 0
    _end = 0
    for tt_s in range(tt_sub):
        tt_d = tt_data[tt_s]
        tt_clip = len(tt_d)
        # print('tt_data len.....', len(tt_d))
        if tt_clip != 0:
            for tt_c in range(tt_clip):
                # print('tt_c......',tt_c)
                feature = tt_d[tt_c]['feature_arr']  # 193 10 150
                label = tt_d[tt_c]['label_arr']
                dis = tt_d[tt_c]['dis_arr']
                dom = tt_d[tt_c]['primary_arr']
                score = tt_d[tt_c]['score_arr']

                dd_len = feature.shape[0]
                _end = _start + dd_len
                test_data[_start:_end, :, :] = feature
                test_label[_start:_end, :] = label
                test_dis[_start:_end, :] = dis
                test_dom[_start:_end, :] = dom
                test_score[_start:_end, :] = score

                _start = _start + dd_len

    test_data = test_data
    test_label = test_label
    test_dis = test_dis
    test_dom = test_dom
    test_score = test_score
    #lstm o channel
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], channels, 5))
    test_data = np.transpose(test_data, (0, 2, 1, 3))
    print('test_data shape.....:', test_data.shape)
    print('test_score shape.....:', test_score.shape)
    print('test_label shape......', test_label.shape)
    print('test_dom shape......', test_dom.shape)
    test_data = np.reshape(test_data, (test_data.shape[0], time_steps, channels*fea_dim))

    train_data = np.empty((train_v_len, time_steps, 150))
    train_label = np.empty((train_v_len, 9))
    train_dis = np.empty((train_v_len, 9))
    train_dom = np.empty((train_v_len, 9))
    train_score = np.empty((train_v_len, 9))

    t_sub = len(t_data)
    print('train sub.....', t_sub)

    start = 0
    end = 0
    for t_s in range(t_sub):
        t_d = t_data[t_s]
        t_clip = len(t_d)
        if t_clip != 0:
            for t_c in range(t_clip):
                feature = t_d[t_c]['feature_arr'] #193 10 150
                label = t_d[t_c]['label_arr']
                dis = t_d[t_c]['dis_arr']
                dom = t_d[t_c]['primary_arr']
                score = t_d[t_c]['score_arr']

                d_len = feature.shape[0]
                end = start + d_len

                train_data[start:end, :, :] = feature
                train_label[start:end, :] = label
                train_dis[start:end, :] = dis
                train_dom[start:end, :] = dom
                train_score[start:end, :] = score

                start = start + d_len

    train_data = train_data
    train_label = train_label
    train_dis = train_dis
    train_dom = train_dom
    train_score = train_score

    #lstm on channel
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1],channels, 5))
    train_data = np.transpose(train_data,(0,2,1,3))
    print('train_data shape....:',train_data.shape)
    print ('train_score shape....:', train_score.shape)
    print('train_label shape.....', train_label.shape)
    print('train_dom shape.....', train_dom.shape)

    train_data = np.reshape(train_data,(train_data.shape[0],time_steps, channels*fea_dim))


    return train_data, test_data, train_dis, test_dis, train_dom, test_dom



def ldl_loso_new(path1, all_sub, sub_id, db_name):
    sub_num, clip_num, channels, fea_dim = get_num(db_name)
    _, info = get_data_info(path1)
    data = all_sub
    info = info
    sub_id = sub_id
    sub = len(data)

    t_data = []
    train_v_len = 0
    for s in range(sub):
        if s == sub_id:
            test_data = data[s]

            v_list = info[s][2]
            v_num = len(v_list)
            v_len = 0
            for v_n in range(v_num):
                v_len = v_len + v_list[v_n]

            clip = len(test_data)
            if clip != 0:
                for c in range(clip):
                    feature = test_data[c]['feature_arr']
                    label = test_data[c]['label_arr']
                    dis = test_data[c]['dis_arr']
                    score = test_data[c]['score_arr']
                    if c == 0:
                        f_tmp = feature
                        l_tmp = label
                        d_tmp = dis
                        s_tmp = score
                    else:
                        f_tmp = np.vstack((f_tmp, feature))
                        l_tmp = np.vstack((l_tmp, label))
                        d_tmp = np.vstack((d_tmp, dis))
                        s_tmp = np.vstack((s_tmp, score))
                test_data = f_tmp
                test_label = l_tmp
                test_dis = d_tmp
                test_score = s_tmp
                test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1],channels, 5))
                test_data = np.transpose(test_data,(0,2,1,3))
                print('test_data shape:', test_data.shape)
                test_data = np.reshape(test_data,(test_data.shape[0],channels,fea_dim))

        else:

            t_clip = len(data[s])
            if t_clip != 0:
                t_data.append(data[s])
                train_v_list = info[s][2]
                train_v_num = len(train_v_list)

                for t_v_n in range(train_v_num):
                    train_v_len = train_v_len + train_v_list[t_v_n]

    train_data = np.empty((train_v_len, 20, 150))
    train_label = np.empty((train_v_len, 9))
    train_dis = np.empty((train_v_len, 9))
    train_score = np.empty((train_v_len, 9))

    t_sub = len(t_data)

    start = 0
    end = 0
    for t_s in range(t_sub):
        t_d = t_data[t_s]
        t_clip = len(t_d)
        if t_clip != 0:
            for t_c in range(t_clip):
                feature = t_d[t_c]['feature_arr'] #193 10 150
                label = t_d[t_c]['label_arr']
                dis = t_d[t_c]['dis_arr']
                score = t_d[t_c]['score_arr']

                d_len = feature.shape[0]
                end = start + d_len

                train_data[start:end, :, :] = feature
                train_label[start:end, :] = label
                train_dis[start:end, :] = dis
                train_score[start:end, :] = score

                start = start + d_len

    train_data = train_data
    train_label = train_label
    train_dis = train_dis
    train_score = train_score
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1],channels, 5))
    train_data = np.transpose(train_data,(0,2,1,3))
    print('train_data shape:',train_data.shape)
    print ('train_score shape:', train_score.shape)

    train_data = np.reshape(train_data,(train_data.shape[0],channels,fea_dim))


    train_label = np.reshape(train_label, [-1, ])
    test_label = np.reshape(test_label, [-1, ])

    return train_data, test_data, train_label, test_label, train_dis, test_dis, train_score, test_score

