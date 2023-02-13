from policies import base_policy as bp

import numpy as np

class Group1(bp.Policy):
    
    def __init__(self, policy_args, board_size, stateq, actq, modelq, logq, id, game_duration, score_scope):
        super().__init__(policy_args, board_size, stateq, actq, modelq, logq, id, game_duration, score_scope)
        self.q_table = {}
        self.epsilon = 0.05
        self.decay = 0.1
        self.window_size = 3
        self.learning_rate = 0.1
        self.discount_rate = 0.9
    
    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        pass
    
    def calculate_Q():
        
        

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.log(f"learning in round {round}")

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        a =  np.random.choice(bp.Policy.ACTIONS)
        self.log(f"in round {round} usin action {a}" )
        return a
    


