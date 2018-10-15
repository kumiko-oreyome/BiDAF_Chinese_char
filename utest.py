import argparse
import util
import data,core

#parser = argparse.ArgumentParser()
#parser.add_argument('--char-dim', default=8, type=int)
#parser.add_argument('--char-channel-width', default=5, type=int)
#parser.add_argument('--char-channel-size', default=100, type=int)
#parser.add_argument('--context-threshold', default=400, type=int)
#parser.add_argument('--dev-batch-size', default=100, type=int)
#parser.add_argument('--dev-file', default='dev-v1.1.json')
#parser.add_argument('--dropout', default=0.2, type=float)
#parser.add_argument('--epoch', default=100, type=int)
#parser.add_argument('--exp-decay-rate', default=0.999, type=float)
#parser.add_argument('--gpu', default=0, type=int)
#parser.add_argument('--hidden-size', default=100, type=int)
#parser.add_argument('--learning-rate', default=0.5, type=float)
#parser.add_argument('--print-freq', default=5, type=int)
#parser.add_argument('--train-batch-size', default=60, type=int)
#parser.add_argument('--train-file', default='train-v1.1.json')
#parser.add_argument('--word-dim', default=100, type=int)
#args = parser.parse_args()
#
#

def test_load_args_from_config():
    util.config_to_arg(args,'config')
    print(args)


def test_preprocessing():
    data.charspan_drcd_preprocessing('./datas/5.json','./datas/5.jsonl')
    data.charspan_drcd_preprocessing('./datas/20.json','./datas/20.jsonl')


def test_load_dataset():
    field = data.SpanFieldCollection()
    loader = data.SpanDataLoader('./datas/20.jsonl',field)
    field.get_word_vector()
    print(len(loader.dataset.examples))

def test_create_vocab():
    field = data.SpanFieldCollection()
    train_loader = data.SpanDataLoader('./datas/20.jsonl',field)
    dev_loader = data.SpanDataLoader('./datas/5.jsonl',field)
    field.build_vocab(train_loader.dataset,dev_loader.dataset)
    print(len(field.get_vocab().itos))


def test_create_model():
    from model.model import BiDAFModelFactory
    argument = core.Argument()
    field = data.SpanFieldCollection()
    train_loader = data.SpanDataLoader('./datas/20.jsonl',field)
    dev_loader = data.SpanDataLoader('./datas/5.jsonl',field)
    field.build_vocab(train_loader.dataset,dev_loader.dataset)
    print(len(field.get_word_vocab().itos)) 


    #pretrained =  self.exp.dsloader.get_word_vectors()
        #if pretrained is None:
            #pretrained = torch.randn(args.word_vocab_size,args.word_dim,device=)
    factory = BiDAFModelFactory()
    model = factory.create_model(None,**(argument.get_model_args(len(field.get_char_vocab()),len(field.get_word_vocab()))))
    print(model.get_hypers())



def test_train():
    pass



    
#test_create_model()
#test_load_dataset()
#test_create_vocab()
#test_preprocessing()
#test_load_args_from_config()
#def test_split():
#    pass
#
#
#def test_load_batch():
#     Experiment('charspan_small',None,refresh_all=True)
#
#
#
#def test_load_exp():
#    exp = Experiment('charspan_small',None,refresh_all=True)
#    exp.build()
#
#def test_drcd_charspan_loader():
#    #FakeArg = namedtuple('FakeArg', 'context_threshold train_batch_size dev_batch_size gpu')
#    #args = FakeArg(0,32,32,-1)
#    args.context_threshold =0
#    args.train_batch_size =32
#    args.dev_batch_size = 32
#    args.gpu = -1
#    exp = Experiment('charspan_small',args,refresh_all=True)
#    exp.build()
    



#test_load_exp()
#test_drcd_charspan_loader()