#!/bin/bash

vector_size=40

# num_of_communities=5
# for (( i=1; i<= $num_of_communities+1; i++))
# do
#     echo "$i of $num_of_communities"
#     python3 review_linear_regr_community.py $vector_size $i $num_of_communities
# done
# 
# num_of_communities=10
# for (( i=1; i<= $num_of_communities+1; i++))
# do
#     echo "$i of $num_of_communities"
#     python3 review_linear_regr_community.py $vector_size $i $num_of_communities
# done

num_of_communities=15
for (( i=1; i<= $num_of_communities+1; i++))
do
    echo "$i of $num_of_communities"
    python3 review_linear_regr_community.py $vector_size $i $num_of_communities
done

num_of_communities=20
for (( i=1; i<= $num_of_communities+1; i++))
do
    echo "$i of $num_of_communities"
    python3 review_linear_regr_community.py $vector_size $i $num_of_communities
done
