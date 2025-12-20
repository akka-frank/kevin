# 电影推荐系统
基于 MovieLens 数据集构建的 "召回 - 排序" 两阶段电影推荐系统，通过 ItemCF 协同过滤挖掘用户兴趣，结合 LightGBM 与 XGBoost 实现精准排序，解决海量电影资源下的个性化推荐问题。

系统架构
    
    召回层：融合 3 种策略（ItemCF 近期兴趣、ItemCF 长期兴趣、热门电影），从 9724 部电影中筛选 50 个候选集；

    排序层：基于多维度特征训练双模型，对候选集精准排序，输出 Top10 推荐结果；

核心功能

    数据预处理：支持 MovieLens 数据集加载、按用户划分8:2 训练 / 测试集，含文件缺失、格式错误等异常处理；

    探索性数据分析（EDA）：自动生成评分分布、用户 / 电影评分数量分布可视化图表；

    特征工程：构建电影特征（类型独热编码、上映年份、评分统计）与用户特征（评分行为统计）；

    混合召回：ItemCF 算法捕捉用户近期 / 长期兴趣，热门电影兜底新用户冷启动场景；

    双模型排序：实现 LightGBM 与 XGBoost 排序模型

    性能对比：量化输出两种模型的排序质量、召回能力与训练效率，提供选型依据


运行环境及依赖：

    Python 3.11.7
    pandas 2.3.3
    numpy 1.26.4
    scipy 1.11.4
    matplotlib 3.8.0
    seaborn 0.13.2
    scikit-learn 1.2.2
    lightgbm 4.6.0
    xgboost 3.0.1

复现步骤：

    环境准备：安装 Python 3.11.7，执行 pip install pandas==2.3.3 numpy==1.26.4 scipy==1.11.4 scikit-learn==1.2.2 lightgbm==4.6.0 xgboost==3.0.1 matplotlib==3.8.0 seaborn==0.13.2。
    
    数据集准备：下载数据集，将 ratings.csv 和 movies.csv 放入当前目录。
    
    运行代码：直接执行 python main.py 一键运行全流程，可以在main函数中修改要推荐的用户。
