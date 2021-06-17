import torch
import loss_functions
import json


from pathlib import Path

def create_path_to_file(file_name,until=-1):
    Path('/'.join(file_name.split('/')[:until])).mkdir(parents=True,
                                                    exist_ok=True)


def parameters_to_str(d):
    return json.dumps(d,separators=(',', ':'))


def BPR_train_original(dataset, neural_network, loss_class, epoch, neg_k=1):
    bpr: loss_functions.BPRLoss = loss_class

    # with timer(name="Sample"):
    # S = utils.UniformSample_original(dataset)
    S = dataset
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users,
                            posItems,
                            negItems,
                            batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        # if world.tensorboard:
        # w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    # time_info = timer.dict()
    # timer.zero()
    return f"loss{aver_loss:.3f}"
