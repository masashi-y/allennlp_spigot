#!/bin/bash

vocab=./vocabulary
pretrained="/home/masashi-y/my_spigot/biaffine_parser_mini.tar.gz"

conf="wo_pos_convergence"
types="dm psd pas"

for type in $types; do
    train=/home/masashi-y/$type/data/english/english_${type}_augmented_train.sdp
    dev=/home/masashi-y/$type/data/english/english_${type}_augmented_dev.sdp
    test_id=/home/masashi-y/$type/data/english/english_id_${type}_augmented_test.sdp
    test_ood=/home/masashi-y/$type/data/english/english_ood_${type}_augmented_test.sdp
    base_test_id=`basename $test_id`
    base_test_ood=`basename $test_ood`

    if [ $type = "dm" ]; then
        device=0
    elif [ $type = "psd" ]; then
        device=1
    else
        device=2
    fi

    env vocab=$vocab  \
        train=$train  \
        dev=$dev  \
        test=$test_id  \
        device=$device  \
        pretrained=""  \
        allennlp train  \
        --force  \
        --include-package spigot  \
        --serialization-dir /work/masashi-y/spigot_${type}_${conf}  \
        configs/syntactic_then_semantic_dependencies_wo_pos_converge.jsonnet  &

    # env vocab=$vocab  \
    #     train=$train  \
    #     dev=$dev  \
    #     test=$test_id  \
    #     device=1  \
    #     pretrained=$pretrained  \
    #     allennlp train  \
    #     --include-package spigot  \
    #     --serialization-dir /work/masashi-y/spigot_${type}_freezed_${conf}  \
    #     configs/syntactic_then_semantic_dependencies_wo_pos.jsonnet  &
    # 
done

wait



# for type in $types; do
#     train=/home/masashi-y/$type/data/english/english_${type}_augmented_train.sdp
#     dev=/home/masashi-y/$type/data/english/english_${type}_augmented_dev.sdp
#     test_id=/home/masashi-y/$type/data/english/english_id_${type}_augmented_test.sdp
#     test_ood=/home/masashi-y/$type/data/english/english_ood_${type}_augmented_test.sdp
#     base_test_id=`basename $test_id`
#     base_test_ood=`basename $test_ood`
#     for freeze in "" "_freezed"; do
#         allennlp predict  \
#             --include-package spigot  \
#             --use-dataset-reader  \
#             --cuda-device 1  \
#             --predictor semantic_dependencies_predictor  \
#             --silent  \
#             --output /work/masashi-y/spigot_${type}${freeze}_${conf}/$base_test_id  \
#             /work/masashi-y/spigot_${type}${freeze}_${conf}/model.tar.gz  \
#             $test_id &
# 
#         allennlp predict  \
#             --include-package spigot  \
#             --use-dataset-reader  \
#             --cuda-device 2  \
#             --predictor semantic_dependencies_predictor  \
#             --silent  \
#             --output /work/masashi-y/spigot_${type}${freeze}_${conf}/$base_test_ood  \
#             /work/masashi-y/spigot_${type}${freeze}_${conf}/model.tar.gz  \
#             $test_ood &
# 
#         wait
# 
#         echo "###" > /tmp/tmp
#         cat /work/masashi-y/spigot_${type}${freeze}_${conf}/$base_test_id >> /tmp/tmp
#         mv /tmp/tmp /work/masashi-y/spigot_${type}${freeze}_${conf}/$base_test_id
# 
#         cd toolkit
#         sh run.sh Scorer  \
#             $test_id.unaugmented  \
#             /work/masashi-y/spigot_${type}${freeze}_${conf}/$base_test_id  \
#             representation=${type}  \
#             2>> /work/masashi-y/spigot_${type}${freeze}_${conf}/$base_test_id.results
#         cd ..
# 
#         echo "###" > /tmp/tmp
#         cat /work/masashi-y/spigot_${type}${freeze}_${conf}/$base_test_ood >> /tmp/tmp
#         mv /tmp/tmp /work/masashi-y/spigot_${type}${freeze}_${conf}/$base_test_ood
# 
#         cd toolkit
#         sh run.sh Scorer  \
#             $test_ood.unaugmented  \
#             /work/masashi-y/spigot_${type}${freeze}_${conf}/$base_test_ood  \
#             representation=${type}  \
#             2>> /work/masashi-y/spigot_${type}${freeze}_${conf}/$base_test_ood.results
#         cd ..
#     done
# done
# 
# 
