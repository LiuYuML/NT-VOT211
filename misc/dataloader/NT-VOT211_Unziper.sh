#!/bin/bash

# 指定zip文件的路径
ZIP_FILE_PATH="/path/to/NT-VOT211.zip"

# 指定解压的目标路径
TARGET_PATH="/path/to/target"

# 创建目标文件夹，如果它不存在的话
mkdir -p "$TARGET_PATH/NT-VOT211"

# 解压文件到目标文件夹
unzip -o "$ZIP_FILE_PATH" -d "$TARGET_PATH/NT-VOT211"

# 检查解压是否成功
if [ $? -eq 0 ]; then
    echo "解压成功"
else
    echo "解压失败"
fi