#!/bin/bash

# 获取当前时间戳并格式化为年月日小时分钟
timestamp=$(date "+%Y-%m%d-%H%M%S")

# 拉取最新代码
git pull

# 添加所有更改到暂存区
git add --all

# 提交更改，commit信息包含当前时间戳
git commit -m "Update at $timestamp"

# 推送到远程仓库
git push
