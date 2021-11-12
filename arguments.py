"""
	This file contains the arguments to parse at command line.
	File main.py will call get_args, which then the arguments
	will be returned.
"""
import argparse

def get_args() -> dict:
	""" Parses arguments at command line.

		Returns:
			args (dict): the arguments parsed
	"""
	parser = argparse.ArgumentParser()

    # Can be 'train' or 'test'
	parser.add_argument('--mode', dest='mode', type=str, default='train')
    # Actor model filename              
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')
    # critic model filename     
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   

	args = parser.parse_args()

	return args