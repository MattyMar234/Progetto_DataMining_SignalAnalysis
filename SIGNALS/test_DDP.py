import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from torchvision import datasets, transforms

# Modello molto semplice
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_ddp(rank, world_size, master_addr, master_port):
    print(f"Avvio rank {rank} su {master_addr}:{master_port}")
    setup(rank, world_size, master_addr, master_port)

    torch.manual_seed(0)
    device = torch.device(f"cuda:0")

    # Dataset distribuito
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Modello
    model = Net().to(device)
    ddp_model = DDP(model, device_ids=[0])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training
    for epoch in range(3):
        sampler.set_epoch(epoch)  # molto importante!
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = ddp_model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            if batch % 100 == 0 and rank == 0:  # solo il master stampa
                print(f"[Epoch {epoch} Batch {batch}] Loss: {loss.item()}")

    cleanup()

def run_ddp(world_size, master_addr, master_port):
    mp.spawn(demo_ddp,
             args=(world_size, master_addr, master_port),
             nprocs=1,
             join=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, help="ID del nodo (0=master, 1=worker)")
    parser.add_argument("--world_size", type=int, default=2, help="Numero totale di nodi")
    parser.add_argument("--master_addr", type=str, required=True, help="IP del nodo master")
    parser.add_argument("--master_port", type=str, default="12355", help="Porta di comunicazione")
    args = parser.parse_args()

    # Ogni nodo esegue con il suo rank
    demo_ddp(args.rank, args.world_size, args.master_addr, args.master_port)
