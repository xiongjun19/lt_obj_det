# coding=utf8
"""
本文件主要实现批量训练mmdetection的脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from mmcv import Config, DictAction
import train_single


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create a file handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)


class BatchTrainer(object):
    def __init__(self, cfg):
        self.config = cfg

    def _get_all_running_configs(self):
        _trained_models = set()
        if not self.config.overwrite:
            _trained_models = self._load_trained_files()
        res = set()
        _dir = self.config.config_dir
        f_pat_arr = self._get_file_pats()
        for f_pat in f_pat_arr:
            f_path_arr = self._find_file_by_pat(f_pat, _dir)
            res.update(f_path_arr)
        f_res = []
        for f_path in res:
            model_name = os.path.basename(f_path)[:-3]
            if model_name not in _trained_models:
                f_res.append(f_path)
        return f_res

    def _find_file_by_pat(self, f_pat, _dir):
        res = []
        for path in Path(_dir).rglob(f_pat):
            f_path = str(path)
            if f_path.endswith(".py"):
                res.append(f_path)
        return res

    def _get_file_pats(self):
        pat_str = self.config.f_pat
        if pat_str is None:
            return ["*.py"]
        pat_arr = [_pat.strip() for _pat in pat_str.split(",")]
        return pat_arr

    def run(self):
        running_cfg_arr = self._get_all_running_configs()
        self._log_cfg_files(running_cfg_arr)
        for cfg_f in running_cfg_arr:
            self.config.config = cfg_f
            train_single.main(self.config)

    def _log_cfg_files(self, cfg_arr):
        logger.info(f"the following files would be trained in this process")
        for cfg_f in cfg_arr:
            logger.info(f"config file_name is: {cfg_f}")

    def _log_trained_files(self, f_arr):
        logger.info(f"the following files are already trained before")
        for cfg_f in f_arr:
            logger.info(f"trained model_name is: {cfg_f}")

    def _load_trained_files(self):
        res = set()
        work_dir = getattr(self.config, "model_dir", "work_dirs")
        if not os.path.exists(work_dir):
            return res
        model_name_arr = os.listdir(work_dir)
        for base_name in model_name_arr:
            if self._is_trained(os.path.join(work_dir, base_name)):
                res.add(base_name)
        self._log_trained_files(res)
        return res

    def _is_trained(self, dir_name):
        """
        依据是否有完成模型文件来判断是否训练过了
        :param dir_name:
        :return:
        """
        f_arr = os.listdir(dir_name)
        for f_name in f_arr:
            if f_name.endswith(".pth"):
                return True
        return False


def main(cfg):
    trainer = BatchTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, help="the directory of config files")
    parser.add_argument("--f_pat", type=str, default=None,
                        help="the pattern of the interested files, if set None, all files will be trained")
    parser.add_argument("--overwrite", action='store_true',
                        help="if set true, then the model that's been trained would not be trained again")

    parser.add_argument("--model-dir", type=str, default='work_dirs', help='the dir to save logs and models')
    # copy the config from official training script
    parser.add_argument("--work-dir", type=str, help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file (deprecate), '
             'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    main(args)
