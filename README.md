# Mixer-SNN

## Run
```commandline
bash terminal.sh
```

## terminal.sh
1. CIFAR-10
```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --T 4 --batch-size 64 --model mixer_conv_encode --epochs 300 --output-dir ./logs --lr 0.1 --lr-scheduler step --lr-step-size 40 --lr-gamma 0.5 --data cifar10 --opt sgd --lr-warmup-epochs 0 --exp-name cifar10
```
2. ImageNet
```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --T 4 --batch-size 64 --model mixer_conv_encode --epochs 300 --output-dir ./logs --lr 0.1 --lr-scheduler step --lr-step-size 40 --lr-gamma 0.5 --data imagenet --data-path /defaultShare/ILSVRC2012/images --opt sgd --lr-warmup-epochs 0 --exp-name imagenet
```

## Tensorboard
```commandline
tensorboadr --logdir=./logs
```

## Model Config
*models/configs.py*
```python
def get_mixer_conv_encode_config():
    config = ml_collections.ConfigDict()
    config.name = 'mixer_conv_encode'
    config.img_size = 224 # input image size
    config.patch_size = 16 # patch size, related to dimension S with (S, C) feature map
    config.encode_dim = 32 # encode layer
    config.hidden_dim = 768 # related to dimension C with (S, C) feature map
    config.token_hidden_dim = 256 # token hidden dim in token mixing block
    config.channel_hidden_dim = 2048 # channel hidden dim in channel mixing block
    config.num_blocks = 4 # number of mixer blocks
    return config
```


## Arguments
```python
parser.add_argument('--exp-name', default='mixer-exp', type=str)
parser.add_argument('--data', default='cifar10', type=str) # cifar10 / imagenet
parser.add_argument('--data-path', default='./data', type=str)
parser.add_argument('--model', default='mixer_conv_encode', type=str)
parser.add_argument('--T', default=4, type=int) # time window
parser.add_argument('--cupy', action='store_true')  # one way to accelerate neuron
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--batch-size', default=32, type=int)   # batch size
parser.add_argument('--epochs', default=90, type=int)   # train epochs
parser.add_argument('--workers', default=16, type=int)  # dataloader workers
parser.add_argument('--opt', default='sgd', type=str)   # optimizer
parser.add_argument('--lr', default=0.1, type=float)    # learning rate
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=0., type=float)
parser.add_argument('--betas', default=[0.9, 0.999], type=float, nargs=2)
parser.add_argument('--criterion', default='ce', type=str)  # cross-entropy loss
parser.add_argument('--lr-scheduler', default='cosa', type=str)
parser.add_argument('--lr-warmup-epochs', default=10, type=int)
parser.add_argument('--lr-warmup-method', default='linear', type=str)
parser.add_argument('--lr-warmup-decay', default=0.01, type=float)
parser.add_argument('--lr-step-size', default=30, type=int)
parser.add_argument('--lr-gamma', default=0.1, type=float)
parser.add_argument('--output-dir', default='./logs', type=str) # log output dir
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--world-size', default=1, type=int)    # preserve
parser.add_argument('--dist-url', default='env://', type=str)   # preserve
parser.add_argument('--seed', default=42, type=int) # set seed
parser.add_argument('--amp', action='store_true')   # auto mix precision
parser.add_argument('--clip-grad-norm', default=None, type=float)
parser.add_argument("--local_rank", type=int)   # preserve
parser.add_argument('--clean', action='store_true')
parser.add_argument('--record-fire-rate', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--label-smoothing', type=float, default=0.0)
parser.add_argument('--fine-tune', default=None, type=str)
```