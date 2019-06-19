# from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import sys
import pickle as p
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse as sp


def save(obj, fname):
    with open(fname, 'wb') as f:
        p.dump(obj, f)


def _parse_posts(file, tagName):
    tree = ET.parse(file)
    dict_obj = {}
    for element in tree.getiterator(tagName):
        idx = element.get('Id', '')
        element = element.attrib
        element.pop('Id', None)
        dict_obj[idx] = element
    return dict_obj


def _parse_users(file, tagName):
    tree = ET.parse(file)
    dict_obj = {}
    for element in tree.getiterator(tagName):
        idx = element.get('Id', '')
        element = element.attrib
        element.pop('Id', None)
        dict_obj[idx] = element
    return dict_obj


def _parse_comments(file, tagName, posts, users):
    tree = ET.parse(file)
    missing_keys = []
    for idx, item in enumerate(tree.getiterator(tagName)):
        item = item.attrib
        postID = item.get("PostId", "")
        text = item.get("Text", "")
        userID = item.get("UserId", "")
        score = item.get("Score", "0")
        if posts.get(postID, "") != "":
            posts[postID]['users'] = posts[postID].get('users', {})
            posts[postID]['users'][userID] = posts[postID]['users'].get(
                userID, 0)+int(score)
    print("Missing %d" % (len(missing_keys)))
    return posts


def _parse_related_posts(file, tagName, posts):
    tree = ET.parse(file)
    missing_keys = []
    for item in tree.getiterator(tagName):
        item = item.attrib
        postID = item.get("PostId", "")
        relatedpostID = item.get("RelatedPostId", "")
        linkType = item.get("LinkTypeId", "")
        if linkType == "1":
            if posts.get(relatedpostID, "") == "":
                posts[relatedpostID] = {"Body": "", 'relatedLink': []}
            try:
                posts[postID]['relatedLink'] = posts[postID].get(
                    'relatedLink', [])
                posts[postID]['relatedLink'].append(relatedpostID)
            except Exception as e:
                missing_keys.append(e)
    print("Missing %d" % (len(missing_keys)))
    return posts


def _get_stats(lbl_mat):
    _lbl_mat = lbl_mat.tocsr()
    _lbl_mat.__dict__['data'][:] = 1
    avg_lbl_per_insta = np.sum(_lbl_mat.sum(axis=1))/(_lbl_mat.shape[0]+1e-3)
    _lbl_mat = _lbl_mat.tocsc()
    avg_insta_per_lbl = np.sum(_lbl_mat.sum(axis=0))/(_lbl_mat.shape[1]+1e-3)
    print("\t#Instances: %d\n\t#Labels: %d" % _lbl_mat.shape)
    print("\tavg lbl/insta: %f\n\tavg insta/lbl: %f," %
          (avg_lbl_per_insta, avg_insta_per_lbl))
    del _lbl_mat


def _create_mat_post2user(posts, users):
    user_idx = dict((k, i)
                    for i, (k, v) in enumerate(users.items()))
    idx_user = dict((i, k)
                    for i, (k, v) in enumerate(users.items()))
    num_labels = len(users.items())
    num_instances = len(list(posts.keys()))
    lbl_mat = lil_matrix((num_instances, num_labels+1), dtype=np.float32)
    fts = []
    keys = []
    for idx, (key, val) in enumerate(posts.items()):
        fts.append(val['Body'])
        keys.append(key)
        p_users = val.get('users', {}).items()
        if len(p_users) > 0:
            _users, _score = zip(
                *(map(lambda x: (user_idx.get(x[0], num_labels), np.exp(x[1]).clip(max=1e5)), p_users)))
            lbl_mat[idx, _users] = _score
        if idx % 10000 == 0:
            print("[%d/%d]" % (idx+1, num_instances), end='\r')
    print("[%d/%d]" % (idx+1, num_instances), end='\r')
    lbl_mat = lbl_mat[:, :-1]
    return lbl_mat, fts, keys, idx_user


def _create_mat_post2post(posts):
    user_idx = dict((k, i)
                    for i, (k, v) in enumerate(posts.items()))
    idx_user = dict((i, k)
                    for i, (k, v) in enumerate(posts.items()))
    num_labels = len(posts.items())
    num_instances = len(list(posts.keys()))
    lbl_mat = lil_matrix((num_instances, num_labels+1), dtype=np.int32)
    fts = []
    keys = []
    for idx, (key, val) in enumerate(posts.items()):
        fts.append(val['Body'])
        keys.append(key)
        p_posts = val.get('relatedLink', [])
        if len(p_posts) > 0:
            _users, _score = zip(
                *(map(lambda x: (user_idx[x], 1), p_posts)))
            lbl_mat[idx, _users] = _score
        if idx % 10000 == 0:
            print("[%d/%d]" % (idx+1, num_instances), end='\r')
    print("[%d/%d]" % (idx+1, num_instances), end='\r')
    lbl_mat = lbl_mat[:, :-1]
    return lbl_mat, fts, keys, idx_user


def _create_y_mat(lbl_mat, fts, keys, idx_user):
    _lbl = lbl_mat.copy().tocsr()
    _lbl.__dict__['data'][:] = 1
    _lbl = _lbl.tocsc()
    valid_lbl = np.where(np.ravel(_lbl.sum(axis=0)) > 1)[0]
    _lbl = _lbl[:, valid_lbl]
    _lbl = _lbl.tocsr()
    _lbl.eliminate_zeros()
    valid_ins = np.where(np.ravel(_lbl.sum(axis=1)) > 0)[0]
    _lbl_mat = lbl_mat.tocsc()[:, valid_lbl]
    _lbl_mat = _lbl_mat.tocsr()[valid_ins, :]
    _fts = list(map(lambda x: fts[x], valid_ins))
    _keys = list(map(lambda x: keys[x], valid_ins))
    _usr_idx = list(map(lambda x: idx_user[x], valid_lbl))
    del _lbl
    return {"Normal": (_fts, _lbl_mat, _keys, _usr_idx)}


def _create_zero_y_mat(lbl_mat, fts, keys, idx_user):
    _lbl = lbl_mat.copy().tocsr()
    _lbl.__dict__['data'][:] = 1
    _lbl = _lbl.tocsc()
    valid_lbl = np.where(np.ravel(_lbl.sum(axis=0)) == 1)[0]
    _lbl = _lbl[:, valid_lbl]
    _lbl = _lbl.tocsr()
    _lbl.eliminate_zeros()
    valid_ins = np.where(np.ravel(_lbl.sum(axis=1)) > 0)[0]
    _lbl_mat = lbl_mat.tocsc()[:, valid_lbl]
    _lbl_mat = _lbl_mat.tocsr()[valid_ins, :]
    _fts = list(map(lambda x: fts[x], valid_ins))
    _keys = list(map(lambda x: keys[x], valid_ins))
    _usr_idx = list(map(lambda x: idx_user[x], valid_lbl))
    del _lbl
    return {"Zero": (_fts, _lbl_mat, _keys, _usr_idx)}


if __name__ == '__main__':
    folder = sys.argv[1]
    out_dir = os.path.join(folder, '')
    posts = os.path.join(folder, 'Posts.xml')
    comments = os.path.join(folder, 'Comments.xml')
    rposts = os.path.join(folder, 'PostLinks.xml')
    users = os.path.join(folder, 'Users.xml')
    print("Parsing Posts")
    posts = _parse_posts(posts, 'row')
    print("Parsing Users")
    users = _parse_users(users, 'row')
    print("Parsing Comments")
    posts = _parse_comments(comments, 'row', posts, users)
    print("Related Posts")
    posts = _parse_related_posts(rposts, 'row', posts)

    print("Building Datasets Post2User")
    lbl_mat, fts, keys, idx_user = _create_mat_post2user(posts, users)
    print("Data stats before")
    _get_stats(lbl_mat)
    print("Building Y dataset")
    Normal = _create_y_mat(lbl_mat, fts, keys, idx_user)
    print("DataStatistics-Y %s:" % (folder))
    _get_stats(Normal["Normal"][1])
    print("Building Zero-Y dataset")
    Zerosh = _create_zero_y_mat(lbl_mat, fts, keys, idx_user)
    print("DataStatistics-Zero-Y %s:" % (folder))
    _get_stats(Zerosh["Zero"][1])
    save(Normal, "%sY_Post2User.pkl" % out_dir)
    save(Zerosh, "%s0_Y_Post2User.pkl" % out_dir)

    print("Building Datasets Post2Post")
    lbl_mat, fts, keys, idx_user = _create_mat_post2post(posts)
    print("Data stats before")
    _get_stats(lbl_mat)
    print("Building Y dataset")
    Normal = _create_y_mat(lbl_mat, fts, keys, idx_user)
    print("DataStatistics-Y %s:" % (folder))
    _get_stats(Normal["Normal"][1])
    print("Building Zero-Y dataset")
    Zerosh = _create_zero_y_mat(lbl_mat, fts, keys, idx_user)
    print("DataStatistics-Zero-Y %s:" % (folder))
    _get_stats(Zerosh["Zero"][1])
    save(Normal, "%sY_Post2Post.pkl" % out_dir)
    save(Zerosh, "%s0_Y_Post2Post.pkl" % out_dir)
