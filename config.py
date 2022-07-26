import argparse
import os
"""
Detailed hyper-parameter configurations.
"""
class Param:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        self.args = all_args

    def all_param(self, parser):
        parser.add_argument("--max_sen_len", default=300, type=int)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--epochs", default=10, type=int)
        parser.add_argument("--is_saved", default=False, type=bool)
        parser.add_argument("--save_processed_data", default=True, type=bool)


        return parser