
import re,json,itertools,os,util
import torch
from torchtext import data
from util import dump_object

def generate_bathiters(path,batch_size,gpu_id,dict_fields):
    dataset = data.TabularDataset(path,format='json',fields=dict_fields)
    return data.BucketIterator(dataset,batch_size,device=gpu_id,sort_key=lambda x: len(x.c_word),)

def dump_as_lines(l,path):
    with open(path,"w",encoding="utf-8") as f:
        for item in l:
            json.dump(item,f,ensure_ascii=False)
            print('',file=f)


# "我是 小明" -->  "我是小明" --> ["我","是","小","明"]
def fake_tokenize(text):
    text = text.replace(" ","")
    return [c for c in text]
    
def charspan_drcd_preprocessing(src_path,tar_path):
    from drcd import RawCorpus
    corpus =  RawCorpus(src_path)
    examples  = corpus.get_rc_examples()
    l = []
    for example in examples:
        spans = find_charspans(example.context,example.answer)
        for start,end in spans:
            l.append({"context":example.context,"answer":example.answer,"qid":example.qid,\
                       "question":example.question,"start":start,"end":end})
    dump_as_lines(l,tar_path)

def find_charspans(text,pattern):
    return [(t.start(), t.end()-1) for t in list(re.finditer(pattern,text))]



class SpanFieldCollection():
    def __init__(self):
        self.RAW = data.RawField()
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=fake_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=fake_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

    #https://discuss.pytorch.org/t/aligning-torchtext-vocab-index-to-loaded-embedding-pre-trained-weights/20878/2
    def load_dataset(self,path,demo=False):
        if not demo:
            self.dict_fields = {
                       'qid': ('id', self.RAW),
                       'start': ('start', self.LABEL),
                       'end': ('end', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}
        else:
            self.dict_fields  = {
                       'qid': ('id', self.RAW),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}
        
        return data.TabularDataset(path,format='json',fields=self.dict_fields)


    def build_vocab(self,word_vectors,*datasets):
        self.CHAR.build_vocab(*datasets)
        if word_vectors is None:
            self.WORD.build_vocab(*datasets)
        else:
            self.WORD.build_vocab(*datasets,vectors=word_vectors)

        
    def get_word_vocab(self):
        return self.WORD.vocab

    def get_char_vocab(self):
        return self.CHAR.vocab

    def get_word_vector(self):
        vocab = self.WORD.vocab
        if hasattr(vocab , 'vectors'):
            return vocab.vectors
        return None

    def dump_fields(self,path):
        util.dump_object((self.CHAR,self.WORD),path,"pkl")

    def load_fileds(self,path):
        self.CHAR,self.WORD = util.load_object(path,"pkl")
        

    



class SpanDataLoader():

    def __init__(self,path,field_collection,context_threshold=-1,demo_flag=False):
        assert os.path.exists(path)
        self.path = path
        self.field_collection = field_collection
        self.context_threshold = context_threshold
        self.demo_flag = demo_flag
        self.load_dataset()


    def load_dataset(self):
        self.dataset = self.field_collection.load_dataset(self.path,self.demo_flag)
        if self.context_threshold > 0:
            self.dataset.examples = [e for e in self.dataset.examples if len(e.c_word) <= self.context_threshold]
    



    def get_batchiter(self,batch_size,gpu_id):
        return SpanIterator(self,batch_size,device_id=gpu_id) 
    
    def get_word_vocab(self):
        return self.field_collection.get_word_vocab()
        
    def get_char_vocab(self):
        return self.field_collection.get_char_vocab()





class SpanIterator():
    def __init__(self,datset_loader,batch_size,device_id,sort_key=lambda x: len(x.c_word)):
        self.datset_loader = datset_loader
        self.device_id = device_id
        self.batch_size = batch_size
        self.dataset = datset_loader.dataset
        self.batchiter = data.BucketIterator(self.dataset,batch_size,device=device_id,sort_key=lambda x: len(x.c_word),repeat=False)
        self.vocab = self.datset_loader.get_word_vocab()

    def get_batch(self):
        return self.batchiter



    def get_dataset(self):
        return self.dataset

    def dump_fields(self,path):
        self.datset_loader.field_collection.dump_fields(path)

    # 這裡設計的不好... 因為batch可能不是這個iter的,所以batch size可能會不一樣
    def decode_batch(self,batch,s_idx,e_idx):
        answers = {}
        for i in range(batch.batch_size):
            id = batch.id[i]
            answer = batch.c_word[0][i][s_idx[i]:e_idx[i] + 1]
            #因為是character based的所以不用用空白隔開
            answer = ''.join([self.get_vocab().itos[idx] for idx in answer])
            answers[id] = answer
        return answers


    def get_vocab(self):
        return self.vocab

    def get_device(self):
        return torch.device(f"cuda:{self.device_id}" if (torch.cuda.is_available() and self.device_id >=0) else "cpu")







    


