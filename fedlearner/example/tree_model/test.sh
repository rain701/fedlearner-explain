#!/bin/bash

set -ex

cd "$( dirname "${BASH_SOURCE[0]}" )"

rm -rf exp data
#生成数据，输出类型为 tfrecord
python make_data.py --verify-example-ids=1 --dataset=iris --output-type=tfrecord
#同时运行follower和leader，训练模型，并输出模型结果
python -m fedlearner.model.tree.trainer follower \
    --verbosity=1 \
    --local-addr=localhost:50052 \
    --peer-addr=localhost:50051 \
    --verify-example-ids=true \
    --file-ext=.tfrecord \
    --file-type=tfrecord \
    --data-path=data/follower_train.tfrecord \
    --validation-data-path=data/follower_test \
    --checkpoint-path=exp/follower_checkpoints \
    --cat-fields=f00001 \
    --output-path=exp/follower_train_output.output &

python -m fedlearner.model.tree.trainer leader \
    --verbosity=1 \
    --local-addr=localhost:50051 \
    --peer-addr=localhost:50052 \
    --verify-example-ids=true \
    --file-ext=.tfrecord \
    --file-type=tfrecord \
    --data-path=data/leader_train.tfrecord \
    --validation-data-path=data/leader_test \
    --checkpoint-path=exp/leader_checkpoints \
    --cat-fields=f00001 \
    --output-path=exp/leader_train_output.output

wait
#同时运行leader和follower来测试模型，并输出结果
python -m fedlearner.model.tree.trainer leader \
    --verbosity=1 \
    --local-addr=localhost:50051 \
    --peer-addr=localhost:50052 \
    --mode=test \
    --verify-example-ids=true \
    --file-type=tfrecord \
    --file-ext=.tfrecord \
    --data-path=data/leader_test/ \
    --cat-fields=f00001 \
    --load-model-path=exp/leader_checkpoints/checkpoint-0004.proto \
    --output-path=exp/leader_test_output &

python -m fedlearner.model.tree.trainer follower \
    --verbosity=1 \
    --local-addr=localhost:50052 \
    --peer-addr=localhost:50051 \
    --mode=test \
    --verify-example-ids=true \
    --file-type=tfrecord \
    --file-ext=.tfrecord \
    --data-path=data/follower_test/ \
    --cat-fields=f00001 \
    --load-model-path=exp/follower_checkpoints/checkpoint-0004.proto \
    --output-path=exp/follower_test_output

wait


rm -rf exp data
#生成数据，数据文件格式为csv
python make_data.py --dataset=iris --verify-example-ids=1 --output-type=csv
#除数据格式变化外，和上面没有区别
python -m fedlearner.model.tree.trainer follower \
    --verbosity=1 \
    --local-addr=localhost:50052 \
    --peer-addr=localhost:50051 \
    --file-type=csv \
    --data-path=data/follower_train.csv \
    --cat-fields=f00001 \
    --checkpoint-path=exp/follower_checkpoints \
    --output-path=exp/follower_train_output.output &
#leader 忽略两个特征
python -m fedlearner.model.tree.trainer leader \
    --verbosity=1 \
    --local-addr=localhost:50051 \
    --peer-addr=localhost:50052 \
    --file-type=csv \
    --data-path=data/leader_train.csv \
    #忽略的特征字段
    --ignore-fields=f00000,f00001 \
    --checkpoint-path=exp/leader_checkpoints \
    --output-path=exp/leader_train_output.output

wait

python -m fedlearner.model.tree.trainer follower \
    --verbosity=2 \
    --local-addr=localhost:50052 \
    --peer-addr=localhost:50051 \
    --mode=test \
    --file-type=csv \
    --data-path=data/follower_test/ \
    #不同
    --cat-fields=f00001 \
    --load-model-path=exp/follower_checkpoints/checkpoint-0004.proto \
    --output-path=exp/follower_test_output &

python -m fedlearner.model.tree.trainer leader \
    --verbosity=2 \
    --local-addr=localhost:50051 \
    --peer-addr=localhost:50052 \
    --mode=test \
    #预测不需要数据?
    --no-data=true \
    --load-model-path=exp/leader_checkpoints/checkpoint-0004.proto \
    --output-path=exp/leader_test_output

wait
#合并预测结果
python merge_scores.py \
    --left-data-path=data/follower_test/ \
    --left-file-ext=.csv \
    --left-select-fields=example_id \
    --right-data-path=exp/leader_test_output \
    --right-file-ext=.output \
    --right-select-fields=prediction \
    --output-path=exp/merge_output
