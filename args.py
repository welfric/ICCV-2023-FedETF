import argparse


def args_parser():
    # no abbreviated flags, main.py matches the full flag names to decide what the command line sets
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Data
    parser.add_argument('--iid', type=int, default=0,
                        help='set 1 for iid, and 0 for noniid (dir. sampling)')
    parser.add_argument('--batchsize', type=int, default=128, 
                        help="batchsize")
    parser.add_argument('--validate_batchsize', type=int, default=128, 
                        help="batchsize")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, 
                    help="dirichlet_alpha")
    parser.add_argument('--scenario', type=str, default='standard',
                        help="{standard, one_class}; one_class: node_num = number of classes, client k trains on every training image of class k "
                             "and tests on the test images of class k and of its nearest class under the centred CLIP geometry (needs the CLIP text embeddings)")

    # System
    parser.add_argument('--device', type=str, default='1',
                        help="cuda device: {cuda, cpu}")
    parser.add_argument('--node_num', type=int, default=20, 
                        help="Number of nodes") 
    parser.add_argument('--T', type=int, default=200, 
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=3, 
                        help="Number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="Type of algorithms:{mnist, cifar10,cifar100, fmnist, tinyimagenet}") 
    parser.add_argument('--select_ratio', type=float, default=1.0,
                    help="the ratio of client selection in each round")
    parser.add_argument('--local_model', type=str, default='CNN',
                        help='Type of local model: {CNN, ResNet20, ResNet18}')
    parser.add_argument('--random_seed', type=int, default=10,
                        help="random seed for the whole experiment")
    parser.add_argument('--exp_name', type=str, default='FirstTable',
                        help="experiment name")

    # Server function
    parser.add_argument('--server_method', type=str, default='fedavg',
                        help="FedAvg, or others")
    parser.add_argument('--server_epochs', type=int, default=20,
                        help="optimizer epochs on server for FedDF etc.")
    parser.add_argument('--server_optimizer', type=str, default='adam',
                        help="type of fusion optimizer, adam or sgd")
    parser.add_argument('--server_valid_ratio', type=float, default=0.02, 
                    help="the ratio of proxy set in the central server")  
                        
    # Client function
    parser.add_argument('--client_method', type=str, default='local_train',
                        help="client method")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--client_valid_ratio', type=float, default=0.3,
                    help="the ratio of validate set in the clients for testing personalization")  
    parser.add_argument('--lr', type=float, default=0.04,  
                        help='learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=5e-4,
                        help='clients local wd rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--mu', type=float, default=0.001,
                        help="proximal term mu")

    # For FedETF
    parser.add_argument('--scaling_train', type=float, default=1.0,
                        help="scaling hyperparameter for training")
    parser.add_argument('--fedetf_anchors', type=str, default='etf',
                        help="fixed classifier of fedetf: {etf, semantic}; semantic uses the FedSem anchors T_lambda (--fedsem_* and --clip_* flags) with everything else unchanged")
    parser.add_argument('--fedetf_loss', type=str, default='balanced',
                        help="local loss of fedetf: {balanced, ce}")
    parser.add_argument('--fedsem_margin', type=float, default=0.0,
                        help="fedetf local training: add margin * max(0, t_y^T t_j) to the logit of every class j != y (no effect with the ETF, whose anchor cosines are all negative)")
    parser.add_argument('--personalize', type=int, default=1,
                        help="set 0 to skip the local finetuning (personalization) after the last round")

    # For FedSem (fixed semantic classifier from CLIP text embeddings)
    parser.add_argument('--fedsem_lambda', type=float, default=0.5,
                        help="blend between the simplex ETF (0) and the centred CLIP geometry (1)")
    parser.add_argument('--fedsem_scale', type=float, default=10.0,
                        help="fixed logit scale s")
    parser.add_argument('--fedsem_shuffle', type=int, default=0,
                        help="set 1 for the shuffled-anchor control (random permutation of the anchors)")
    parser.add_argument('--fedsem_learnable', type=int, default=0,
                        help="set 1 to unfreeze the anchors (learnable-head control)")
    parser.add_argument('--fedsem_loss', type=str, default='ce',
                        help="local loss: {ce, balanced}")
    parser.add_argument('--clip_backend', type=str, default='openai',
                        help="CLIP text encoder backend: {openai, hf}")
    parser.add_argument('--clip_model', type=str, default=None,
                        help="CLIP model name, defaults to ViT-B/32 (openai) or openai/clip-vit-base-patch32 (hf)")
    parser.add_argument('--fedsem_text_emb', type=str, default='',
                        help="cache file for the class-name text embeddings, loaded if it exists and written otherwise")

    args = parser.parse_args()

    return args
