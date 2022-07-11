import torch
from spikingjelly.activation_based import encoding, functional
from spikingjelly.activation_based.model import train_classify, train_imagenet_example

from models.model import MlpMixer
from models.configs import get_mixer_b16_config


class MixerSNNTrainer(train_classify.Trainer):
    def __init__(self):
        super(MixerSNNTrainer, self).__init__()
        self.encoder = encoding.PoissonEncoder()
        self.model_configs = {
            'mixer_b16': get_mixer_b16_config()
        }

    def preprocess_train_sample(self, args, x: torch.Tensor):
        x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
        return self.encoder(x)

    def preprocess_test_sample(self, args, x: torch.Tensor):
        x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
        return self.encoder(x)

    def process_model_output(self, args, y: torch.Tensor):
        return y.mean(0)

    def get_args_parser(self, add_help=True):
        parser = super(MixerSNNTrainer, self).get_args_parser()
        parser.add_argument('--T', type=int, help="total time-steps")
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        parser.add_argument('--model-config', type=str, choices=['mixer_b16'], help='select model')
        return parser

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f'_T{args.T}'

    def load_model(self, args, num_classes):
        model_config = self.model_configs[args.model_config]

        model = MlpMixer(model_config)

        functional.set_step_mode(model, 'm')
        if args.cupy:
            functional.set_backend(model, 'cupy')
        return model


if __name__ == '__main__':
    trainer = MixerSNNTrainer()
    args = trainer.get_args_parser().parse_args()
    trainer.main(args)

# python -m torch.distributed.launch --nproc_per_node=4 train.py --T 4 --model-config mixer_b16 --data-path /defaultShare/ILSVRC2012/images --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90 --train-crop-size 224
# python -m torch.distributed.launch --nproc_per_node=2 -m spikingjelly.activation_based.model.train_imagenet_example --T 4 --model spiking_resnet18 --data-path /defaultShare/ILSVRC2012/images --batch-size 64 --lr 0.1 --lr-scheduler cosa --epochs 90