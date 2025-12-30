# Base
python tracking/train.py --script lorat --config base_224 --save_dir ./output --mode multiple --nproc_per_node 2
python tracking/train.py --script lorat --config base_384 --save_dir ./output --mode multiple --nproc_per_node 2

# Large
python tracking/train.py --script lorat --config large_224 --save_dir ./output --mode multiple --nproc_per_node 2
python tracking/train.py --script lorat --config large_384 --save_dir ./output --mode multiple --nproc_per_node 2

# Giant
python tracking/train.py --script lorat --config giant_224 --save_dir ./output --mode multiple --nproc_per_node 2
python tracking/train.py --script lorat --config giant_378 --save_dir ./output --mode multiple --nproc_per_node 2
