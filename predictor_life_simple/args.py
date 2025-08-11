

class DataloaderArgs:
    train_batch_size: int
    train_num_workers: int
    train_shuffle: bool
    test_batch_size: int
    test_num_workers: int
    test_shuffle: bool

class OptimizerArgs:
    name: str
    args: dict

class LRSchedulerArgs:
    name: str
    args: dict

class TrainingArgs:
    epochs: int

class Args:
    dataloader_args: DataloaderArgs
    optimizer_args: OptimizerArgs
    lr_scheduler_args: LRSchedulerArgs
    training_args: TrainingArgs

