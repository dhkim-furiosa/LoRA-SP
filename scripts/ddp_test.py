import torch
import torch.distributed as dist
import os
import datetime

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=datetime.timedelta(minutes=5)
    )

    torch.cuda.set_device(local_rank)  # set device first
    dist.barrier()  # without device_ids

    # each process creates a tensor on its local GPU
    tensor = torch.ones(1, device=f"cuda:{local_rank}") * rank

    dist.barrier()  # without device_ids

    if rank == 0:
        print(f"[Rank {rank}] broadcasting value {tensor.item()} to all ranks")

    # overwrite all ranks' tensors with rank 0 value
    dist.broadcast(tensor, src=0)

    print(f"[Rank {rank}] after broadcast: {tensor.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()