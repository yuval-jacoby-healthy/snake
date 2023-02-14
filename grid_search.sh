#!/bin/bash



for lr in 0.001
do
    for radius in 3 5
    do
        for epsilon in 0.3333
        do	
            for decay in 1.0
            do
                for ldf in 1 2 3 4 5
                do
                echo "$lr" "$radius" "$epsilon" "$decay" "$ldf"

                Python3 /Users/ariel/Documents/Miscellaneous/snake/Snake.py --policies "Benshapiro(epsilon=$epsilon,lr=$lr,radius=$radius,decay=$decay)" --game_duration 2500 --board_size "(10,100)" --score_scope 500 --log_file "my_log.log" --to_render 0  -rt game.pickle --player_init_time 1
                done
            done
        done
    done
done 
