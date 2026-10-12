
import time
import sys
import torch
from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
import numpy as np
import os
import torch.nn as nn
import copy
import torch.optim as optim
import torch.nn.functional as F
import math
from pyhessian import hessian
from server_funct import *
import wandb
from client_funct import *
from fedsem import build_fedsem_anchors, load_text_emb, semantic_neighbours

if __name__ == '__main__':

    args = args_parser()

    ##### Exp settings #####
    ##### change it for different exps, flags given on the command line take precedence #####
    exp_settings = {
        'client_method': 'fedetf',
        'exp_name': 'NCFL',
        'node_num': 20,
        'iid': 0,
        'random_seed': 10,
        'dirichlet_alpha': 0.05,
        'local_model': 'ResNet20',
        'dataset': 'cifar10',
        'T': 200,
        'E': 3,
    }
    for key, value in exp_settings.items():
        if not any(arg == '--' + key or arg.startswith('--' + key + '=') for arg in sys.argv[1:]):
            setattr(args, key, value)

    assert args.fedetf_anchors in ['etf', 'semantic'], 'Undefined fedetf anchors...'
    assert args.fedetf_loss in ['balanced', 'ce'], 'Undefined fedetf loss...'
    assert args.fedsem_margin >= 0, 'the semantic margin must be >= 0'
    assert args.scale_init is None or args.scale_init > 0, 'the logit scale must be > 0'
    assert args.ntd_beta >= 0 and args.ntd_tau > 0, 'FedNTD needs beta >= 0 and tau > 0'
    assert args.scenario in ['standard', 'one_class'], 'Undefined scenario...'
    # set the server method automatically
    args = set_server_method(args)
    # set the random seed for controlling the randomness
    setup_seed(args.random_seed)


    # setting hyperparameter for feddyn, fedprox, and ditto
    if args.client_method == 'feddyn':
        args.mu = 0.01
    elif args.client_method in ['fedprox', 'ditto']:
        args.mu = 0.001
    
    # hyperparameter for different datasets
    if args.dataset in ['cifar100', 'tinyimagenet']:
        args.lr = 0.01
    
    setting_name =  args.exp_name + '_' + args.dataset + '_' + args.local_model + '_nodenum' + str(args.node_num) + '_dir' + str(args.dirichlet_alpha) +'_E'+ str(args.E)  + '_C' + str(args.select_ratio) \
    + '_' + args.server_method + '_' + args.client_method + '_seed' + str(args.random_seed)
    if args.client_method == 'fedetf' and args.fedetf_anchors == 'semantic':
        setting_name += '_semantic_lambda' + str(args.fedsem_lambda) + ('_shuffled' if args.fedsem_shuffle else '')
    if args.client_method == 'fedetf' and args.fedetf_loss == 'ce':
        setting_name += '_ce'
    if args.client_method == 'fedetf' and args.fedsem_margin > 0:
        setting_name += '_margin' + str(args.fedsem_margin)
    if args.client_method == 'fedetf' and args.ntd_beta > 0:
        setting_name += '_ntd' + str(args.ntd_beta) + '_tau' + str(args.ntd_tau)
    if args.scale_init is not None:
        setting_name += '_s' + str(args.scale_init)
    if not args.scale_learnable:
        setting_name += '_sfixed'
    if args.scenario == 'one_class':
        setting_name += '_oneclass'
    if args.client_method == 'fedsem':
        setting_name += '_lambda' + str(args.fedsem_lambda) + '_s' + str(args.fedsem_scale) + '_' + args.fedsem_loss \
        + ('_shuffled' if args.fedsem_shuffle else '') + ('_learnable' if args.fedsem_learnable else '')

    root_path = '/code_root'
    output_path = 'results/date'

    wandb.init(
        config = args,
        project = 'NCFL',
        name = setting_name , tags = args.exp_name
    )

    # for ccvr
    if args.client_method == 'ccvr':
        args.client_method = 'local_train'
        args.server_method = 'fedavg'
        method_cache = 'ccvr'
    else:
        method_cache = None

    # set GPU device
    # args.device = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # torch.cuda.set_device('cuda:'+args.device)

    data = Data(args)

    sample_size = []
    for i in range(args.node_num): 
        sample_size.append(len(data.train_loader[i]))
    size_weights = [i/sum(sample_size) for i in sample_size]
    print('size-based weights',size_weights)

    # initialize the central node
    # num_id equals to -1 stands for central node
    central_node = Node(-1, data.test_loader[0], data.test_set, args)

    # fedsem: the server builds the anchors T_lambda once (Algorithm 1)
    if args.client_method == 'fedsem':
        anchors = build_fedsem_anchors(args, data.train_set, central_node.num_classes, central_node.model.linear_proto.out_features)
        central_node.model.semantic_classifier.load_anchors(anchors)
    # fedetf with semantic anchors: T_lambda replaces the random ETF, the clients copy it from the server every round
    elif args.client_method == 'fedetf' and args.fedetf_anchors == 'semantic':
        anchors = build_fedsem_anchors(args, data.train_set, central_node.num_classes, central_node.model.proto_classifier.proto.shape[0])
        central_node.model.proto_classifier.load_proto(anchors.T.contiguous().cuda())

    # one_class scenario: client k tests on class k and its nearest class under the centred CLIP geometry
    client_test = {i: None for i in range(args.node_num)}
    if args.scenario == 'one_class':
        neighbours = semantic_neighbours(load_text_emb(args, data.train_set, central_node.num_classes))
        client_test = data.build_one_class_tests(neighbours)
        class_names = list(data.train_set.classes)
        for k, neighbour in enumerate(neighbours):
            print('client {}: trains on {}, tests on {} + {}'.format(k, class_names[k], class_names[k], class_names[neighbour]))
        global_cm_recorder, local_cm_recorder = [], []

    # initialize the client nodes
    client_nodes = {}
    for i in range(args.node_num): 
        client_nodes[i] = Node(i, data.train_loader[i], data.train_set, args, client_test[i]) 
        client_nodes[i].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
        # fedsem: broadcast T_lambda once
        if args.client_method == 'fedsem':
            client_nodes[i].model.semantic_classifier.load_anchors(central_node.model.semantic_classifier.anchors)

    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []

    if args.select_ratio == 1.0:
        select_list_recorder = [[i for i in range(args.node_num)] for _ in range(args.T)]
    else:
        select_list_recorder = torch.load(os.path.join(root_path, 'outputs/0915/','num'+ str(args.node_num)+'_ratio'+str(args.select_ratio)+ '_select_list_recorder.pth'))

    for rounds in range(args.T):
        print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
        print(setting_name)
        # lr_scheduler(rounds, client_nodes, args)

        # for ccvr, last round calibration
        if (rounds == args.T - 1) and method_cache != None:
            args.server_method = method_cache
            args.client_method = method_cache

        # Client selection
        select_list = select_list_recorder[rounds]

        # Local update
        client_nodes, train_loss = Client_update(args, client_nodes, central_node, select_list)
        if args.scenario == 'one_class':
            # local models after local training, on their own class and its neighbour
            local_cm = torch.stack([confusion_matrix(args, client_nodes[i]) for i in select_list])
            local_cm_recorder.append(local_cm)
            avg_client_acc = sum(cm.diag().sum().item() / cm.sum().item() * 100 for cm in local_cm) / len(select_list)
        else:
            avg_client_acc = Client_validate(args, client_nodes, select_list)
        print(args.server_method + args.client_method + ', averaged clients acc is ', avg_client_acc)
        
        # Server aggregation
        central_node = Server_update(args, central_node, client_nodes, select_list, size_weights)
        if args.scenario == 'one_class':
            global_cm = confusion_matrix(args, central_node, which_dataset = 'local')
            global_cm_recorder.append(global_cm)
            acc = global_cm.diag().sum().item() / global_cm.sum().item() * 100
            own = [local_cm[n, i, i].item() / local_cm[n, i].sum().item() * 100 for n, i in enumerate(select_list)]
            neighbour = [local_cm[n, neighbours[i], neighbours[i]].item() / local_cm[n, neighbours[i]].sum().item() * 100 for n, i in enumerate(select_list)]
            print('local models: own class acc {:.2f}, neighbour class acc {:.2f}'.format(np.mean(own), np.mean(neighbour)))
        else:
            acc = validate(args, central_node, which_dataset = 'local')
        print(args.server_method + args.client_method + ', global model test acc is ', acc)

        # Recorder
        test_acc_recorder.append(acc)
        try:
            wandb.log({'trainloss': train_loss}, step = rounds)
            wandb.log({'testacc': acc}, step = rounds)
            wandb.log({'peracc': avg_client_acc}, step = rounds)
        except:
            pass
        # record the acc of the last 5 rounds for final acc
        if rounds >= args.T - 5:
            final_test_acc_recorder.update(acc)
        # for the last round, save the model
        if rounds == args.T - 1:
            torch.save(central_node.model.state_dict(), os.path.join(root_path, output_path, setting_name+'_finalmodel.pth'))
            if 'fedetf' in args.client_method :
                torch.save(central_node.model.proto_classifier.proto, os.path.join(root_path, output_path, setting_name+'_etfproto.pth'))
            elif args.client_method == 'fedsem':
                torch.save(central_node.model.semantic_classifier.anchors, os.path.join(root_path, output_path, setting_name+'_fedsemanchors.pth'))

    print('final_testacc', final_test_acc_recorder.value())

    # Local finetuning: personalization process
    if args.personalize:
        select_list = [idx for idx in range(len(client_nodes))]
        client_nodes, train_loss = Client_personalization(args, client_nodes, central_node, select_list)
        avg_client_acc = Client_validate(args, client_nodes, select_list)
        print(args.server_method + args.client_method + ', personalization acc is ', avg_client_acc)
    else:
        avg_client_acc = None

    # save the final result
    final_results = {'final_testacc': final_test_acc_recorder.value(), 'final_peracc': avg_client_acc}
    if args.scenario == 'one_class':
        torch.save({'class_names': class_names, 'neighbours': neighbours, 'global_cm': torch.stack(global_cm_recorder),
                    'local_cm': torch.stack(local_cm_recorder)}, os.path.join(root_path, output_path, setting_name+'_confusion.pth'))
    try:
        wandb.log(final_results)
    except:
        pass
    torch.save(final_results, os.path.join(root_path, output_path, setting_name+'_finalresults.pth'))
    torch.save(test_acc_recorder, os.path.join(root_path, output_path, setting_name+'_recorder.pth'))