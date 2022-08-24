import os
import torch
import torch.distributed as dist
from collections import defaultdict, deque
import time
import datetime
import spikingjelly.visualizing
import matplotlib.pyplot as plt
import numpy as np


def init_distributed_mode(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def cal_fire_rate(s_seq):
    return torch.mean(s_seq, dim=0)


def plot_eval_fire_rate(eval_result_path):
    eval_result = torch.load(eval_result_path)
    layers_fr = eval_result['fr_records']
    each_layer_avg_fr = []
    for layer in layers_fr:
        each_layer_avg_fr.append(torch.mean(layers_fr[layer]).cpu().item())
        print(f'{layer} avg fr: ', "%.2f" % torch.mean(layers_fr[layer]).cpu().item())
    print(f'Global avg fr: ', "%.2f" % np.mean(each_layer_avg_fr))

    # x = range(len(layers_fr.keys()))
    # plt.bar(x, each_layer_avg_fr)
    # plt.xlabel('layer index')
    # plt.ylabel('fire rate avg')
    # plt.title('average fire rate of different layers')
    # plt.show()
    #
    # feature_map = layers_fr['module.model.7.model.1.mlp.2'].cpu().numpy()
    # print(feature_map)
    # s, c = feature_map.shape
    # print(s, c)
    # plt.imshow(feature_map, cmap='coolwarm', origin='upper', aspect="auto")
    # plt.colorbar()
    # plt.show()

    def plot_histogram_one_block(block_name_prefix):
        token_mixing_block_lif_fm_1 = layers_fr[f'{block_name_prefix}.model.1.mlp.2'].cpu().numpy()
        token_mixing_block_lif_fm_2 = layers_fr[f'{block_name_prefix}.model.1.lif'].cpu().numpy()
        channel_mixing_block_lif_fm_1 = layers_fr[f'{block_name_prefix}.model.3.mlp.2'].cpu().numpy()
        channel_mixing_block_lif_fm_2 = layers_fr[f'{block_name_prefix}.model.3.lif'].cpu().numpy()

        def plot_one_ax(fm, mean_axis, x_dim, ax, title_prefix):
            fm = np.mean(fm, axis=mean_axis)
            x = range(x_dim)
            zero_fr_num = np.sum(fm <= 0.02)
            print(np.min(fm))
            ax.set_title(f'{title_prefix}_inactive_neuron={zero_fr_num}')
            ax.bar(x, fm)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, constrained_layout=True, dpi=1200)
        plot_one_ax(token_mixing_block_lif_fm_1, 0, token_mixing_block_lif_fm_1.shape[1], ax1, 'token_1')
        plot_one_ax(token_mixing_block_lif_fm_2, 1, token_mixing_block_lif_fm_2.shape[0], ax2, 'token_2')
        plot_one_ax(channel_mixing_block_lif_fm_1, 0, channel_mixing_block_lif_fm_1.shape[1], ax3, 'channel_1')
        plot_one_ax(channel_mixing_block_lif_fm_2, 1, channel_mixing_block_lif_fm_2.shape[0], ax4, 'channel_2')
        fig.show()

    plot_histogram_one_block('module.model.7')



def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def fine_tune_state_dict(model_state_dict, fine_tune_layer):
    model_state_dict['model.14.weight'] = fine_tune_layer.weight
    model_state_dict['model.14.bias'] = fine_tune_layer.bias
    return model_state_dict


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if print_freq > 0 and i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


if __name__ == '__main__':
    plot_eval_fire_rate("./eval_result.pth")