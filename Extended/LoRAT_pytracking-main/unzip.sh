#!/bin/bash

# 定义压缩文件所在的路径
SOURCE_FOLDER="/root/autodl-tmp/train_data"

# 定义解压后的目标路径
TARGET_FOLDER="/root/autodl-tmp/datasets/got10k/train"

# 检查目标文件夹是否存在，如果不存在则创建
if [ ! -d "$TARGET_FOLDER" ]; then
    mkdir -p "$TARGET_FOLDER"
    echo "目标文件夹 $TARGET_FOLDER 已创建。"
fi

# 循环解压文件
for i in $(seq -f "%02g" 1 19); do
    # 构造文件名
    FILE_NAME="GOT-10k_Train_split_$i.zip"
    
    # 构造完整的文件路径
    FULL_FILE_PATH="$SOURCE_FOLDER/$FILE_NAME"
    
    # 检查文件是否存在
    if [ -f "$FULL_FILE_PATH" ]; then
        # 解压文件到目标文件夹
        unzip -o "$FULL_FILE_PATH" -d "$TARGET_FOLDER"
        echo "已解压 $FULL_FILE_PATH 到 $TARGET_FOLDER"
    else
        echo "文件 $FULL_FILE_PATH 不存在，跳过。"
    fi
done

echo "解压完成。"