#!/bin/bash

python3 Snake.py --policies "Avoid(epsilon=0.5)" --game_duration 1000 --score_scope 500 --log_file "my_log.log" --to_render 0  -rt game.pickle --player_init_time 2

# after running that we can reply
python3 Snake.py -p game.pickle -pir 200 -pfr 300 -pit 0 --to_render 1


