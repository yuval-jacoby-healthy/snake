from policies import base_policy as bp

import numpy as np

class Group1(bp.Policy):
    
    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        pass

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.log(f"learning in round {round}")

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        a =  np.random.choice(bp.Policy.ACTIONS)
        self.log(f"in round {round} usin action {a}" )
        return a

