import json,re,random,torch,itertools,os,importlib,pickle
import jieba as jb
import nltk




def vocab_decode(vocab,id_list):
    return ' '.join([vocab.itos[idx] for idx in id_list])


def dump_object(obj,path,form="json"):
    if form == "json":
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj,f,ensure_ascii=False)
    elif  form == "pkl":
        with open(path, 'wb') as f:
            pickle.dump(obj,f)
    else:
        assert False


def load_object(path,form="json"):
    if form == "json":
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
    elif form == "pkl":
        with open(path, 'rb') as f:
            obj = pickle.load(f)     
    else:
        assert False

    return obj

def check_then_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def concat_path(p1,p2):
    return '%s/%s'%(p1,p2)
#TODO優化
#目前會建立兩個corpus .. 太慢









