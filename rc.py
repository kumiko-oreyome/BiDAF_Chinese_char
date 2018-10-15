import torch,json,evaluate,copy,os
from torch import nn, optim
from util import dump_object, check_then_makedir
from tensorboardX import SummaryWriter
from time import gmtime, strftime
from model.ema import EMA
from model.model import BiDAF
from data import SpanFieldCollection









def load_mf_from_run_folder(folder_path):
    model_path = f'{folder_path}/model.pt'
    field_path = f'{folder_path}/field.pkl'
    model = BiDAF.load_model(model_path)
    field = SpanFieldCollection()
    field.load_fileds(field_path)
    return model,field




def create_test_func(evalfunc_name,save_path='tmp.txt'):
    def decorator(evalfunc,save_path):
        def test_on_batch(model,batchiter,answers):
            loss,qid_span_dict = prediction(model,batchiter)
            dump_object(qid_span_dict,save_path)
            results = evalfunc(qid_span_dict,answers)
            return loss, results['exact_match'], results['f1']
        return test_on_batch
    func =    decorator(evaluate.evaluate_rc,save_path)
    return func

    
# loss_criterion : None for not retrun loss
def prediction(model,batchiter,loss_criterion=nn.CrossEntropyLoss()):
    model.eval()
    loss = 0.0
    answers = {}
    device = batchiter.get_device()


    for batch in batchiter.get_batch():
        p1, p2 = model(batch)
        if loss_criterion is not None:
            batch_loss = loss_criterion(p1, batch.start) + loss_criterion(p2, batch.end)
            loss += batch_loss.item()

        # (batch, c_len, c_len)
        batch_size, c_len = p1.size()
        ls = nn.LogSoftmax(dim=1)
        mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
        score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
        score, s_idx = score.max(dim=1)
        score, e_idx = score.max(dim=1)
        
        s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze(1)
        _answers = batchiter.decode_batch(batch,s_idx,e_idx)
        answers.update(_answers)
    
    if loss_criterion is None:
        return answers
    return loss,answers


#TODO save optimizer
def train_by_span(model,train_iter,dev_iter,dev_answers,test_func,epoch_num,learning_rate,exp_decay_rate,save_freq):
    import datetime
    model_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M")
    log_dir = 'runs/' + model_time
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #save fields
    train_iter.dump_fields(f'{log_dir}/field.pkl')
    
    device = train_iter.get_device()
    model = model.to(device)
    #ema = EMA(exp_decay_rate)
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_dir)
    model.train()
    loss= 0
    max_dev_exact, max_dev_f1 = -1, -1
    best_epoch = 0

    for epoch in range(epoch_num):
        print('epoch:%d'%(epoch))
        for i, batch in enumerate(train_iter.get_batch()):
            p1, p2 = model(batch)
            optimizer.zero_grad()
            batch_loss = criterion(p1, batch.start) + criterion(p2, batch.end)
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

            #for name, param in model.named_parameters():
            #    if param.requires_grad:
            #        ema.update(name, param.data)

        
        #backup_params = EMA(0)
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        backup_params.register(name, param.data)
        #        param.data.copy_(ema.get(name))
        dev_loss, dev_exact, dev_f1 = test_func(model,dev_iter,dev_answers)
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        param.data.copy_(backup_params.get(name))

        if (epoch + 1) % save_freq == 0:
            model.dump(f'{log_dir}/model{epoch}.pt')
            if dev_f1 > max_dev_f1:
                max_dev_f1 = dev_f1
                max_dev_exact = dev_exact
                best_model = copy.deepcopy(model)
                best_epoch =  epoch 

        writer.add_scalar('loss/train', loss, epoch)
        writer.add_scalar('loss/dev', dev_loss, epoch)
        writer.add_scalar('exact_match/dev', dev_exact, epoch)
        writer.add_scalar('f1/dev', dev_f1, epoch)
        print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f}'
              f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')

        loss = 0
        model.train()

    writer.close()

    print(f'best epoch:{best_epoch}  max dev EM: {max_dev_exact:.3f} / max dev F1: {max_dev_f1:.3f}')
    best_model.dump(f'{log_dir}/model.pt')
    return best_model    