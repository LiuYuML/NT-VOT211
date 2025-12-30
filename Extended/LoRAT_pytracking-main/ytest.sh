# LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
python tracking/test.py lorat base_224 --dataset lasot --threads 2 --num_gpus 2
python tracking/analysis_results.py # need to modify tracker configs and names

# GOT10K-test
python tracking/test.py lorat base_224 --dataset got10k_test --threads 2 --num_gpus 2
python lib/test/utils/transform_got10k.py --tracker_name lorat --cfg_name base_224_300

# TrackingNet
python tracking/test.py lorat base_224 --dataset trackingnet --threads 2 --num_gpus 2
python lib/test/utils/transform_trackingnet.py --tracker_name lorat --cfg_name base_224_300
# python tracking/test.py lorat giant_378 --dataset ntvot211