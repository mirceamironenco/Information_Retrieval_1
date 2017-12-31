#!/bin/bash
sudo trec_eval -m all_trec -q $1 $2 | grep -E "^(recall_1000|map_cut_1000|ndcg_cut_10 |P_5 ).*all"
