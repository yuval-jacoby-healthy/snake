from collections import defaultdict
from Snake import Game, parse_args

import sys
import os
import numpy as np
import argparse

from policies import base_policy
from policies import *
from policies.base_policy import Policy


def get_random_reward(ocupied):
    n = np.random.randint(6,10)
    while n in ocupied:
        n = np.random.randint(6,10)
    return n

def generateRewardRenderMap():
    # generate rewards
    snake_module = sys.modules["Snake"]
    negative_reward = get_random_reward([])
    
    snake_module.FOOD_RENDER_MAP = {negative_reward : 'X'}
    snake_module.FOOD_VALUE_MAP = {negative_reward : 0}
    snake_module.FOOD_REWARD_MAP = {negative_reward : np.random.randint(-10, 0)}
    
    low_positive_reward = get_random_reward(snake_module.FOOD_RENDER_MAP.keys())
    snake_module.FOOD_RENDER_MAP.update({low_positive_reward : '$'})
    snake_module.FOOD_VALUE_MAP.update({low_positive_reward : 1})
    # snake_module.FOOD_REWARD_MAP.update({low_positive_reward : np.random.randint(0, 10)})
    snake_module.FOOD_REWARD_MAP.update({low_positive_reward : 2})
    
    high_positive_reward = get_random_reward(snake_module.FOOD_RENDER_MAP.keys())
    snake_module.FOOD_RENDER_MAP.update({high_positive_reward : '*'})
    snake_module.FOOD_VALUE_MAP.update({high_positive_reward : 3})
    snake_module.FOOD_REWARD_MAP.update({high_positive_reward : 5})
    
    print(snake_module.FOOD_RENDER_MAP)
    
    

def find_policy(policies_to_find):
    policies = {}
    for module_name in sys.modules:
        if not module_name.startswith('policies.policy'):
            continue    
        # POLICIES = {}
        valid_module = False
        for p in policies_to_find:
            if "policies.policy_" + p == module_name:
                valid_module = True
                
            
        if not valid_module:
            continue
        
        mod = sys.modules[module_name]
        for cls_name in dir(mod):
            try:
                if cls_name != 'Policy':
                    cls = mod.__dict__[cls_name]
                    group_name = module_name[module_name.find("policy_") + len("policy_") : ]
                    if issubclass(cls, Policy):                        
                        policies[group_name] = (cls, {})
            except TypeError:
                pass
    return policies
    
    
def get_game_defaults():
    
    args.__dict__["record_to"] = None
    args.__dict__["playback_from"] = None
    args.__dict__["playback_initial_round"] = None
    args.__dict__["playback_final_round"] = None
    args.__dict__["log_file"] = "game.log"
    args.__dict__["to_render"] = True
    args.__dict__["output_file"] = "game.out"
    args.__dict__["render_rate"] = .1
    
    
# g = p.add_argument_group('Game')
    args.__dict__["board_size"] = (20,60)
    # args.__dict__['board_size'] = [int(x) for x in args.board_size[1:-1].split(',')]
    args.__dict__["obstacle_density"] = .04
    args.__dict__["policy_wait_time"] = .01
    args.__dict__["random_food_prob"] = .2
    args.__dict__["max_item_density"] = .25
    args.__dict__["food_ratio"] = .2
    args.__dict__["game_duration"] = 10000
    args.__dict__["policy_action_time"] = .01
    args.__dict__["policy_learn_time"] = .1
    args.__dict__["player_init_time"] = 10
    args.__dict__["policies"] = None
    
    
    
    
    # g = p.add_argument_group('Players')
    args.__dict__["score_scope"] = 1000
    args.__dict__["init_player_size"] = 5
    
    return args
            
def first_round(policy_to_run, to_render):
    policies = {}
    
    generateRewardRenderMap()
    
    args = get_game_defaults()
    
    args.to_render = to_render
    args.player_init_time = 2
    
    
    args.game_duration = np.random.randint(1000, 2500)
    args.random_food_prob = np.random.uniform(.1, .5)
    args.obstacle_density = np.random.uniform(.05, .1)
    args.board_size = (10, 100) # TODO: Should we change?
    
    policies = find_policy([policy_to_run])
    for name, policy in policies.items():        
        
        args.__dict__['policies'] = [policy]
        args.__dict__['name'] = [name]
        g = Game(args)
        
        g.run()    
    
        
        
def second_round(policies_str, to_render):
    generateRewardRenderMap()
    
    args = get_game_defaults()
    
    args.to_render = to_render
    args.player_init_time = 2
    
    args.game_duration = np.random.randint(1000, 2500)
    args.random_food_prob = np.random.uniform(.1, .5)
    args.obstacle_density = np.random.uniform(.05, .1)
    args.board_size = (10, 100)
    
    
    policies = find_policy(policies_str.split(';'))
    
    args.__dict__['policies'] = list(policies.values())
    args.__dict__['name'] = list(policies.keys())
    g = Game(args)
    g.run()    
    
def get_evalution_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, help="seed for randoms")
    parser.add_argument('--policy', '-p', type=str, help='file name to run seperated by ;')
    parser.add_argument('--round_to_run', '-r', type=int, default=1, help='which round to run')
    parser.add_argument('--to_render', type=int, default=0, help='render each round')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_evalution_args()
    np.random.seed(seed=args.seed)
    
    if (args.round_to_run == 1):
        first_round(args.policy, args.to_render)
    else:
        second_round(args.policy, args.to_render)
    
