import importlib,os,pickle,json,argparse
from util import concat_path
import data
from model.model import BiDAF
import rc
from drcd import RawCorpus
from data import charspan_drcd_preprocessing,SpanDataLoader

# parse args
# preprocessing : path
# load dataset  : path
# create vocab  : *dataset
# create batch  : dataset
# create pretrained : dataset



# create model
# train
# - test on batch
# - evaluate
# - test on example
# - save to directory
# - reload

class Argument():
    def __init__(self):
        self.parse_args()

    def parse_args(self):
        #training
        parser = argparse.ArgumentParser()
        subparsers =  parser.add_subparsers(help='commands',dest='command')
        #common
        parser.add_argument('--config-module-name',default='config')
        parser.add_argument('--batch-size', default=16, type=int)
        parser.add_argument('--gpu_id', default=0, type=int)


        # train
        train_parser = subparsers.add_parser(name='train')
        train_parser.add_argument('--char-dim', default=10, type=int)
        train_parser.add_argument('--char-channel-width', default=1, type=int)
        train_parser.add_argument('--char-channel-num', default=10, type=int)
        train_parser.add_argument('--context-threshold', default=200, type=int)
        train_parser.add_argument('--dev-path', default=None)
        
        train_parser.add_argument('--dropout_rate', default=0.2, type=float)
        train_parser.add_argument('--epoch_num', default=100, type=int)
        train_parser.add_argument('--exp-decay-rate', default=0.999, type=float)
        #parser.add_argument('--hidden-dim', default=10, type=int)
        train_parser.add_argument('--learning-rate', default=0.5, type=float)
        train_parser.add_argument('--save-freq', default=5, type=int)
        train_parser.add_argument('--pretrained-wv-path', default=None, type=str)
        #parser.add_argument('--train-batch-size', default=32, type=int)
        train_parser.add_argument('--train-path', default=None)
        train_parser.add_argument('--word-dim', default=10, type=int)
        train_parser.set_defaults(func=train_pipeline)
        
        #test
        test_parser = subparsers.add_parser(name='test')
        test_parser.add_argument('testpath')
        test_parser.add_argument('runfolder')
        test_parser.set_defaults(func=test_pipeline)
        
        #demo
        demo_parser  = subparsers.add_parser(name='demo')
        demo_parser.add_argument('demopath')
        demo_parser.add_argument('runfolder')
        demo_parser.set_defaults(func=demo_pipeline)


        self.args = parser.parse_args()
        #print(self.args)
        # TODO
        self.load_arg_from_config()

    def load_arg_from_config(self):
        self.config_to_arg(self.args.config_module_name)

    def config_to_arg(self,module_name):
        config = importlib.import_module(module_name)
        self.args.name = config.NAME
        self.args.data_root_path = './datas'
        #args.root_path = concat_path('./exps',exp_name)
        self.args.description = config.DESCRIPTION
        self.args.train_path = concat_path(self.args.data_root_path,config.TRAIN_FILE)
        self.args.dev_path = concat_path(self.args.data_root_path,config.DEV_FILE)
        #self.args.test_path = concat_path(self.args.data_root_path,config.TEST_FILE)

        self.args.pre_train_path = concat_path(self.args.data_root_path,config.PRE_TRAIN_FILE)
        self.args.pre_dev_path = concat_path(self.args.data_root_path,config.PRE_DEV_FILE)
        #self.args.pre_test_path = concat_path(self.args.data_root_path,config.PRE_TEST_FILE)

        self.args.char_channel_num = config.CHAR_CHANNEL_NUM
        self.args.word_dim = config.WORD_DIM
    

    def get_model_args(self,field):
        return {"char_vocab_size":len(field.get_char_vocab()),"char_dim":self.args.char_dim,"char_channel_num":self.args.char_channel_num,"char_channel_width":self.args.char_channel_width,\
                "word_vocab_size":len(field.get_word_vocab()),"word_dim":self.args.word_dim,"dropout_rate":self.args.dropout_rate}

def train_pipeline(args):
    field = data.SpanFieldCollection()
    train_loader = data.SpanDataLoader(args.pre_train_path,field)
    dev_loader = data.SpanDataLoader(args.pre_dev_path,field)
    field.build_vocab(None,train_loader.dataset,dev_loader.dataset)
    model =   BiDAF( **argument_wrapper.get_model_args(field))
    if args.pretrained_wv_path is not None:
        pretrained = load_pretrained_wv(args.pretrained_wv_path)
        model.set_word_embedding(pretrained)
    train_iter = train_loader.get_batchiter(args.batch_size,args.gpu_id)
    dev_iter = dev_loader.get_batchiter(args.batch_size,args.gpu_id)
    test_func = rc.create_test_func('foo')
    best_model = rc.train_by_span(model,train_iter,dev_iter,RawCorpus(args.dev_path).get_answers(),test_func,\
                     args.epoch_num,args.learning_rate,args.exp_decay_rate,args.save_freq)

    best_model.dump('model.pt')




def test_pipeline(args):
    charspan_drcd_preprocessing(args.testpath,'testtmp.jsonl')
    model,field = rc.load_mf_from_run_folder(args.runfolder)
    test_loader = SpanDataLoader('testtmp.jsonl',field)
    test_func = rc.create_test_func('foo')
    testiter = test_loader.get_batchiter(16,args.gpu_id)
    print('data preparation complete')
    model = model.to(testiter.get_device())
    loss,em,fl= test_func(model,testiter,RawCorpus(args.testpath).get_answers())
    print('EM:%.3f F1:%.3f loss:%.3f'%(em,fl,loss))
    os.remove('testtmp.jsonl')



def demo_pipeline(args):
    print('demo %s'%(args.demopath))
    print('read:')
    with open(args.demopath,"r",encoding="utf-8") as f:
        s=f.read()
        obj = json.loads(s,encoding="utf-8")
        print(obj["context"])
    model,field = rc.load_mf_from_run_folder(args.runfolder)
    test_loader = SpanDataLoader(args.demopath,field,demo_flag=True)
    test_iter = test_loader.get_batchiter(2,args.gpu_id)
    model = model.to(test_iter.get_device())
    print('question')
    print(obj["question"])

    print('answer')
    print(rc.prediction(model,test_iter,loss_criterion=None))


def load_pretrained_wv(path):
    return None

argument_wrapper = Argument()
args =  argument_wrapper.args
args.func(args)
