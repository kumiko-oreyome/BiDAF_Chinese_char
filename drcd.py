import json,re,random,torch,itertools,os
import jieba as jb
import nltk



#TODO優化
#目前會建立兩個corpus .. 太慢
def split_drcd_file(src_path,tar1_path,tar2_path,num1,num2,shuffle=False):

    def examples_to_rc_json(corpus,examples,path):
        ids = [example.qid for example in examples]
        corpus.filter_by_qa_pairs(ids)
        corpus.dump(path)
    
    corpus1 = RawCorpus(src_path)
    #這個FUNCITON會改變corpus
    examples  = corpus1.get_rc_examples()
    print(len(examples))
    if shuffle :
        import random
        random.shuffle(examples,random.random)   
    corpus1.filter_by_qa_pairs( [example.qid for example in  examples[:num1+num2]])
    if num1>0:
        examples_to_rc_json(corpus1,examples[:num1],tar1_path)
    if num2 > 0:
        corpus2 = RawCorpus(src_path)
        examples_to_rc_json(corpus2,examples[num1:num1+num2],tar2_path)






    



#def generate_widespan(context,qid,question,answer):
#    context_words = list(jb.cut(context))
#    charpos2wid = []
#    cnt = 0
#    for word in context_words:
#        charpos2wid.extend([cnt]*len(word))
#        cnt+=1
#    
#    question_words = list(jb.cut(question))
#    occurs = [(t.start(), t.end()-1) for t in list(re.finditer(answer,context))]
#    ans_texts = []
#    word_spans = []
#    char_spans = []
#    #cstart,end:character position
#    for cstart,cend in occurs:
#        ans_texts.append("".join([context_words[i]  for i in range(charpos2wid[cstart],charpos2wid[cend]+1)]))
#        word_spans.append((charpos2wid[cstart],charpos2wid[cend]))
#        char_spans.append((cstart,cend))
#    example = DRCDWideSpanRCExample(" ".join(context_words),qid," ".join(question_words),answer,ans_texts,word_spans,char_spans)
#
#    return example




## 改成統一介面執行會變慢因為像是文章的斷詞每個QAㄉ都要重做



class RCCorpus():
    def __init__(self,path=None):
        self.path = path
        self.articles = []
        if path is not None:
            self.parse_articles()


    def dump(self,target_path):
        with open(target_path,'w',encoding='utf-8') as f:
            print('dump %d wide span examples'%len(self.articles))
            json.dump({'data':[article.to_json() for article in self.articles]},f,ensure_ascii=False)

    def parse_articles(self):
        with open(self.path,'r',encoding='utf-8') as f:
            raw_data = json.load(f)['data']   
        for i,article in enumerate(raw_data):
            if i%20==0:
                print('%dth example is loaded'%(i))
            paragraphs = self.parse_paragraphs(article["paragraphs"])
            article = DRCDArticle(article["title"],article["id"],paragraphs)
            self.articles.append(article)

    def parse_paragraphs(self,paragraphs):
        ret = []
        for paragraph in paragraphs:
            context = paragraph["context"]
            pid = paragraph["id"]
            rc_examples = self.parse_qas(context,pid,paragraph["qas"])
            paragraph = DRCDParagraph(context,pid,rc_examples)
            ret.append(paragraph)
        return ret

    def parse_qas(self,context,pid,qas):
        l = []
        for qa in qas:
            for ans in qa['answers']:
                l.append(self.parse_qa_pair(context,qa['id'],qa['question'],ans['text']))
        return l
    
    def parse_qa_pair(self,context,qid,question,answer):
        pass


    def filter_by_qa_pairs(self,qa_ids):
        for article in self.articles:
            tmp1 =[]
            for paragraph in article.paragraphs:
                tmp2 = []
                for qa_pair in paragraph.rc_examples:
                    if qa_pair.qid  in qa_ids:
                        tmp2.append(qa_pair)
                paragraph.rc_examples = tmp2
                if len(paragraph.rc_examples) >0:
                    tmp1.append(paragraph)
            article.paragraphs = tmp1
        self.articles = list(filter(lambda x:len(x.paragraphs)>0,  self.articles))
        
    def get_rc_examples(self):
        return  list(itertools.chain(*[ article.get_rc_examples() for article in self.articles]))

class RawCorpus(RCCorpus):
    def __init__(self,path):
        super().__init__(path)
        
    def parse_qa_pair(self,context,qid,question,answer):
        return  DRCDRawRCExample(context,qid,question,answer)

    def get_answers(self):
        examples = self.get_rc_examples()
        return { example.qid:[example.answer] for example in examples}
    

class  DRCDArticle():
    def __init__(self,title,_id,paragraphs):
        self.title = title
        self.id = _id
        self.paragraphs = paragraphs

    def get_rc_examples(self):
        return  list(itertools.chain(*[ p.get_rc_examples() for p in self.paragraphs]))

    def to_json(self):
        return {'title':self.title,'id':self.id,
        'paragraphs':[p.to_json() for p in self.paragraphs]}


class DRCDParagraph():
    def __init__(self,context,pid,rc_examples):
        self.context = context
        self.pid = pid
        self.rc_examples = rc_examples

    def get_rc_examples(self):
        return self.rc_examples

    def to_json(self):
        l = [example.get_attributes() for example in self.rc_examples]
        context = "" if len(self.context)==0 else l[0]['context']
        qas = []
        for qa in l:
            qas.append({"id":qa["qid"],"question":qa["question"],"answers":[{"id":1,"text":qa["answer"]}]})
        return {"context":context,"id":self.pid,"qas":qas}
                  
#example有分為 給model的example和從原始資料轉換過後的example
# example --> context,pair(model需要的形式交給各個example去做)
# paragraphs 裡面的每一個dict object

class DRCDRawRCExample():
    def __init__(self,context,qid,question,answer):
        self.context = context
        self.qid = qid
        self.question = question
        self.answer = answer

    def get_attributes(self):
        return {'context':self.get_context(),"qid":self.qid,\
                "question":self.get_question(),"answer":self.get_answer()}

    def get_context(self):
        return self.context
    def get_question(self):
        return self.question

    def get_answer(self):
        return self.answer