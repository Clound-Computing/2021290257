import os
import sys
import torch
import argparse
import time
import math
import json
import sklearn
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from functools import wraps
from _thread import start_new_thread
from dgl.data import register_data_args
from torch.nn.parallel import DistributedDataParallel
from GNN_model_architecture import FakeNewsModel
from GNN_evaluation import do_evaluation
from training_helper_functions import get_train_mask_nids, get_features_given_blocks, get_features_given_graph
from inference_operator_helper_functions import compute_articles_dict
import dgl
warnings.filterwarnings('once')

def deregister_torch_ipc():
    from multiprocessing.reduction import ForkingPickler
    import torch
    ForkingPickler._extra_reducers.pop(torch.cuda.Event)
    for t in torch._storage_classes:
        ForkingPickler._extra_reducers.pop(t)
    for t in torch._tensor_classes:
        ForkingPickler._extra_reducers.pop(t)
    ForkingPickler._extra_reducers.pop(torch.Tensor)
    ForkingPickler._extra_reducers.pop(torch.nn.parameter.Parameter)

def running_code(proc_id, n_gpus, args, devices, overall_graph):
    device = devices[proc_id]
    print("Device " + str(device))
    sys.stdout.flush()

    # the port we are currently running on -> used to handle distributed training
    if args.curr_port is not None:
        curr_port_tru = str(args.curr_port)
    else:
        curr_port_tru = '25295'
    world_size = n_gpus

    g = overall_graph._g[0]
    print("Loaded the graph")
    sys.stdout.flush()

    labels_dict = {}
    corpus_df = pd.read_csv(overall_graph.dataset_corpus, sep='\t')
    sources_in_corpus = []
    for index, row in corpus_df.iterrows():
        sources_in_corpus.append(row['source_url'])
        labels_dict[row['source_url_normalized']] = (row['fact'], row['bias'])

    with open(os.path.join(args.path_where_data_is, "data/acl2020", f"splits.json")) as the_file:
        data_splits = json.load(the_file)

    n_nodes_sources = g.number_of_nodes(ntype='source')
    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()
    n_nodes_users = g.number_of_nodes(ntype='user')
    n_nodes_articles = g.number_of_nodes(ntype='article')
    n_edges_articles_talkers = g.number_of_edges(etype='has_talker')
    print("""----Data statistics------'
      #Edges %d
      #Nodes %d
      #Source Nodes %d
      #User Nodes %d
      #Article Nodes %d
      #Article Talker Edges %d""" %
          (n_edges, n_nodes, n_nodes_sources, n_nodes_users, n_nodes_articles, n_edges_articles_talkers))
    sys.stdout.flush()

    out_features = 3
    user_embedding_size = 773
    article_embedding_size = 768

    torch.cuda.set_device(device)

    model = FakeNewsModel(in_features={'source': 778, 'user': user_embedding_size, 'article': article_embedding_size},
                          hidden_features=args.hidden_features, out_features=out_features,
                          canonical_etypes=g.canonical_etypes, num_workers=args.num_workers, n_layers=args.n_layers,
                          conv_type='gcn')
    if torch.cuda.is_available():
        model = model.to(device)

    model = DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    if args.load_model:
        print("Loading the model")
        model.module.load_state_dict(torch.load(args.path_to_save_model))
        model.eval()
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100,
                                                              verbose=True, min_lr=1e-4)
    dur = []
    best_acc = 0.0

    em_options_to_run = args.em_options_to_run_list.split(':')
    em_options_to_run = [int(x) for x in em_options_to_run]
    print("These are the EM options we are going to run")
    print(em_options_to_run)

    if args.run_multiple_data_splits:
        all_the_data_splits_we_want_to_run_now = data_splits.keys()
    else:
        all_the_data_splits_we_want_to_run_now = args.data_splits_key_to_run_when_doing_one.split(':')
    print("Running these many data splits " + str(all_the_data_splits_we_want_to_run_now))

    best_test_acc_average = []
    best_test_acc_at_dev_average = []
    path_to_save_model_old = args.path_to_save_model

    iterations_before_em = 500

    for curr_data_split_key in all_the_data_splits_we_want_to_run_now:
        print("Data split is " + str(curr_data_split_key))
        em_edges_added = {}
        args.path_to_save_model = path_to_save_model_old + curr_data_split_key
        print(args.path_to_save_model)
        print("Running " + str(curr_data_split_key))

        model = FakeNewsModel(
            in_features={'source': 778, 'user': user_embedding_size, 'article': article_embedding_size},
            hidden_features=args.hidden_features, out_features=out_features, canonical_etypes=g.canonical_etypes,
            num_workers=args.num_workers, n_layers=args.n_layers, conv_type='gcn')
        model = model.to(torch.device('cuda'))
        model = DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100,
                                                                  verbose=True, min_lr=1e-4)
        best_acc = 0.0
        best_dev_acc = 0.0
        bes_acc_at_dev = 0.0
        best_train_acc = 0.0
        print("Working with data split " + str(curr_data_split_key))
        curr_data_split = data_splits[curr_data_split_key]

        training_set_to_use = curr_data_split['train']
        if not os.path.isfile(args.path_where_graph_is + '/dev_set_' + curr_data_split_key + '.npy'):
            training_set_to_use, dev_set_to_use = sklearn.model_selection.train_test_split(curr_data_split['train'],
                                                                                           test_size=args.percentage_of_dev_to_use)
            np.save(args.path_where_graph_is + '/dev_set_' + curr_data_split_key + '.npy', dev_set_to_use)
            np.save(args.path_where_graph_is + '/train_set_' + curr_data_split_key + '.npy', training_set_to_use)
        else:
            dev_set_to_use = np.load(args.path_where_graph_is + '/dev_set_' + curr_data_split_key + '.npy')
            training_set_to_use = np.load(args.path_where_graph_is + '/train_set_' + curr_data_split_key + '.npy')
        print("Length of the new train set: " + str(len(training_set_to_use)))
        print("Length of the new dev set: " + str(len(dev_set_to_use)))
        sys.stdout.flush()

        graph_style = 'm2'

        train_mask, dev_mask, test_mask, train_nids, dev_nids, test_nids = get_train_mask_nids(args, overall_graph,
                                                                                               training_set_to_use,
                                                                                               curr_data_split,
                                                                                               dev_set_to_use,
                                                                                               graph_style=graph_style,
                                                                                               use_dev_set=True,
                                                                                               curr_data_split_key=curr_data_split_key)
        train_mask_tensor = torch.from_numpy(train_mask)
        train_idx = torch.nonzero(train_mask_tensor).squeeze()
        test_mask_tensor = torch.from_numpy(test_mask)
        test_idx = torch.nonzero(test_mask_tensor).squeeze()
        dev_mask_tensor = torch.from_numpy(dev_mask)
        dev_idx = torch.nonzero(dev_mask_tensor).squeeze()
        train_labels_idx = None

        curr_g = overall_graph._g[0]

        negative_sampler_to_use = dgl.dataloading.negative_sampler.Uniform(5)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
        train_nid = torch.split(train_idx, math.ceil(len(train_idx) / n_gpus))[proc_id]
        test_nid = torch.split(test_idx, math.ceil(len(test_idx) / n_gpus))[proc_id]
        dataloader_nc = dgl.dataloading.DistNodeDataLoader(curr_g, {'source': train_nid}, sampler, batch_size=64,
                                                           shuffle=True, drop_last=False, num_workers=args.num_workers)
        test_sampler = sampler
        dataloader_test = dgl.dataloading.DistNodeDataLoader(curr_g, {'source': test_nid}, test_sampler, batch_size=64,
                                                             shuffle=True, drop_last=False,
                                                             num_workers=args.num_workers)

        torch.cuda.synchronize(device)
        since = time.time()
        torch.cuda.synchronize(device)
        best_acc = 0.0
        best_train_acc = 0.0

        for epoch in range(args.n_epochs):
            model.train()
            train_labels_idx = None
            train_loss = 0.0
            for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader_nc):
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    feature_dict = get_features_given_blocks(input_nodes, curr_g, blocks)
                    label_tensor = feature_dict['source'][:output_nodes['source'].shape[0], 768:].float().to(device)
                    logits = model(feature_dict, blocks)['source']
                    loss = F.cross_entropy(logits, label_tensor.argmax(dim=1))
                    train_loss += loss.item()
                    if train_labels_idx is None:
                        train_labels_idx = label_tensor.argmax(dim=1)
                    else:
                        train_labels_idx = torch.cat((train_labels_idx, label_tensor.argmax(dim=1)), 0)
                    loss.backward()
                    optimizer.step()
                if step % 100 == 0 and proc_id == 0:
                    print("Loss at step %d is %f" % (step, loss.item()))
            curr_loss = train_loss / len(train_idx)

            model.eval()
            eval_acc = 0.0
            with torch.no_grad():
                for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader_test):
                    feature_dict = get_features_given_blocks(input_nodes, curr_g, blocks)
                    label_tensor = feature_dict['source'][:output_nodes['source'].shape[0], 768:].float().to(device)
                    logits = model(feature_dict, blocks)['source']
                    eval_acc += (logits.argmax(dim=1) == label_tensor.argmax(dim=1)).sum().item()
            eval_acc /= len(test_idx)

            if eval_acc > best_acc:
                best_acc = eval_acc
                print("New best eval acc at epoch %d: %f" % (epoch, eval_acc))
                torch.save(model.module.state_dict(), args.path_to_save_model + "_%d.pth" % epoch)
                print("Model saved")
                sys.stdout.flush()

    print("Finished Training")
    sys.stdout.flush()

def run_multi_proc(args, devices, n_gpus, overall_graph):
    mp.set_start_method('spawn', force=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(running_code, args=(n_gpus, args, devices, overall_graph), nprocs=n_gpus, join=True)

class SomeGraphDataStructure:
    def __init__(self, path_where_data_is):
        self.dataset_corpus = os.path.join(path_where_data_is, "corpus.csv")
        self._g = self.load_graph(os.path.join(path_where_data_is, "graph.bin"))

    def load_graph(self, path):
        return dgl.load_graphs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake news detection')
    parser.add_argument("--curr_port", type=str, default=None,
                        help="The port currently running on")
    parser.add_argument("--path_where_data_is", type=str, default='/mnt/data/siqi_datasets',
                        help="The path where the dataset is located")
    parser.add_argument("--path_where_graph_is", type=str, default='/mnt/data/siqi_datasets',
                        help="The path where the graph is located")
    parser.add_argument("--path_to_save_model", type=str, default='/mnt/data/saved_models/',
                        help="The path where the model is saved")
    parser.add_argument("--hidden_features", type=int, default=64,
                        help="The number of hidden features in the model")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="The number of workers to use for data loading")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="The number of layers in the model")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="The learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="The weight decay")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="The number of epochs to train")
    parser.add_argument("--em_options_to_run_list", type=str, default='1:2:3',
                        help="The EM options to run")
    parser.add_argument("--run_multiple_data_splits", action='store_true',
                        help="Whether to run multiple data splits")
    parser.add_argument("--data_splits_key_to_run_when_doing_one", type=str, default='',
                        help="The data splits key to run when doing one")
    parser.add_argument("--percentage_of_dev_to_use", type=float, default=0.1,
                        help="The percentage of the dev set to use")
    parser.add_argument("--load_model", action='store_true',
                        help="Whether to load a pre-trained model")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    devices = list(range(n_gpus))
    overall_graph = SomeGraphDataStructure(args.path_where_data_is)  # This should be your graph loading code
    run_multi_proc(args, devices, n_gpus, overall_graph)
