from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]


@RunIf(sh=True, min_gpus=1)
@pytest.mark.slow
def test_experiments(tmp_path: Path) -> None:
    """Test running all available experiment configs with `fast_dev_run=True.`

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        # "hydra.sweep.dir=" + str(tmp_path),
        "paths.data_dir=${paths.root_dir}/data",
        "trainer.max_epochs=1",
        "++trainer.limit_train_batches=1",
        "++trainer.limit_val_batches=1",
        "++trainer.limit_test_batches=1",
        "data.batch_size=1",
        # "callbacks.model_checkpoint.every_n_train_steps=1",
        # "callbacks.model_checkpoint.train_time_interval=1",
        "callbacks.model_checkpoint.every_n_epochs=1",
        *overrides,
    ]
    run_sh_command(command)
