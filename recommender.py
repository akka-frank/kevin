import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import random
import lightgbm as lgb
import xgboost as xgb 
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

warnings.filterwarnings('ignore')

"""
部分库的版本：
pandas                            2.3.3
numpy                             1.26.4
scipy                             1.11.4
scikit-learn                      1.2.2
lightgbm                          4.6.0
"""


plt.rcParams['font.family'] = 'SimHei'

# ==================== 1. 问题背景和描述 ====================
class MovieRecommender:
    
    """
    电影推荐系统类
    基于用户历史评分，构建混合推荐系统，包括召回和排序两个阶段
    这个类中有：
        1.load_data方法：读取数据
        2.data_split方法：对于评分数据按照用户划分，确保每个用户看的电影有80%在训练集，有20%在测试集用于验证
        3.exploratory_analysis方法：对数据进行可视化分析
        4.feature_engineering方法：特征工程，输入训练数据集和电影数据
        5.RecallSystem类：内部封装三种召回方式，对外提供混合召回的接口
        6.class RankingModel类：提供排序模型的训练，评估等方法
    """
    
    def __init__(self):
        """初始化推荐系统"""
        self.user_history = None
        self.topk_similarities = None
        self.hot_items = None
        self.movies = None
        self.user_features = None
        
    # ==================== 2. 数据加载与异常处理 ====================
    def load_data(self, rating_path = "ratings.csv", movies_path = "movies.csv"):
        """加载数据，包含异常处理"""
        try:
            rating = pd.read_csv(rating_path)
            movies = pd.read_csv(movies_path)
            rating = rating.sort_values('timestamp')
            print("数据加载成功！")
            print(f"评分数据形状: {rating.shape}")
            print(f"电影数据形状: {movies.shape}")
            return rating, movies
        except FileNotFoundError as e:
            print(f"文件未找到错误: {e}")
            raise
        except Exception as e:
            print(f"数据加载错误: {e}")
            raise
    
    def data_split(self, rating, movies, test_ratio=0.2):
        """数据分割函数，按用户划分训练集和测试集"""
        try:
            data_user_movie = rating.groupby("userId")["movieId"].apply(list).reset_index()
            
            def split_user_history(movies_list, ratio=0.8):
                split = max(1, int(len(movies_list) * ratio))
                return movies_list[:split], movies_list[split:]
            
            data_user_movie[['train', 'test']] = data_user_movie['movieId'].apply(
                lambda x: pd.Series(split_user_history(x, 1-test_ratio))
            )
            
            train_interactions = (
                data_user_movie[['userId', 'train']]
                .explode('train')
                .rename(columns={'train': 'movieId'})
                .dropna()
            )
            
            test_interactions = (
                data_user_movie[['userId', 'test']]
                .explode('test')
                .rename(columns={'test': 'movieId'})
                .dropna()
            )
            
            train_data = train_interactions.merge(
                rating[['userId', 'movieId', 'rating']], on=["userId", "movieId"], how="left"
            ).merge(movies, on="movieId", how="left")
            
            test_data = test_interactions.merge(
                rating[['userId', 'movieId', 'rating']], on=["userId", "movieId"], how="left"
            ).merge(movies, on="movieId", how="left")
            
            return train_data, test_data
            
        except Exception as e:
            print(f"数据分割错误: {e}")
            raise
    
    # ==================== 3. 探索性数据分析 ====================
    def exploratory_analysis(self, rating, movies):
        """探索性数据分析"""
        print("\n=== 探索性数据分析 ===")
        
        # 数据基本信息
        print("1. 数据基本信息:")
        print(f"用户数量: {rating['userId'].nunique()}")
        print(f"电影数量: {rating['movieId'].nunique()}")
        print(f"评分数量: {len(rating)}")
        print(f"评分时间范围: {rating['timestamp'].min()} 到 {rating['timestamp'].max()}")
        
        # 评分分布可视化
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        rating['rating'].hist(bins=10, edgecolor='black')
        plt.title('评分分布')
        plt.xlabel('评分')
        plt.ylabel('频率')
        
        plt.subplot(1, 3, 2)
        user_rating_counts = rating.groupby('userId').size()
        user_rating_counts[user_rating_counts < 100].hist(bins=50, edgecolor='black')
        plt.title('用户评分数量分布')
        plt.xlabel('评分数量')
        plt.ylabel('用户数量')
        
        plt.subplot(1, 3, 3)
        movie_rating_counts = rating.groupby('movieId').size()
        movie_rating_counts[movie_rating_counts < 100].hist(bins=50, edgecolor='black')
        plt.title('电影评分数量分布')
        plt.xlabel('评分数量')
        plt.ylabel('电影数量')
        
        plt.tight_layout()
        plt.show()
        
    
    # ==================== 4. 特征工程 ====================
    def feature_engineering(self, train_data, movies):
        """特征工程"""
        # 电影类型独热编码
        genres = movies['genres'].str.split("|", expand=True).stack().unique()
        for genre in genres:
            movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)
        
        # 提取电影年份
        movies["movie_year"] = movies['title'].str.extract(r'\((\\d{4})\)')
        movies["movie_year"] = movies["movie_year"].fillna(0).astype("int32")
        
        # 电影统计特征
        movie_features = train_data.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std'],
            'userId': 'nunique'
        }).round(3)
        movie_features.columns = ['movie_avg_rating', 'movie_rating_count', 
                                 'movie_rating_std', 'movie_unique_users']
        movie_features['movie_rating_std'] = movie_features['movie_rating_std'].fillna(0)
        movie_features = movie_features.reset_index()
        
        # 合并电影特征
        movies = movies.merge(movie_features, on="movieId", how="left")
        
        # 用户特征
        user_features = train_data.groupby('userId').agg({
            'rating': ['mean', 'count', 'std'],
            'movieId': 'nunique'
        }).round(3)
        user_features.columns = ['user_avg_score', 'user_rating_count', 
                                'user_rating_std', 'user_unique_movies']
        user_features = user_features.reset_index()
        
        self.movies = movies
        self.user_features = user_features
        
        return movies, user_features
    
    # ==================== 5. 召回系统（基于ItemCF） ====================
    class RecallSystem:
        """召回系统类，包含多种召回策略"""
        def __init__(self, user_history, topk_similarities, hot_items):
            self.user_history = user_history
            self.topk_similarities = topk_similarities
            self.hot_items = hot_items
            self.recent_last_n = 10
            self.long_term_threshold = 50
            self.hot_top_n = 200
            self.recall_num = 50
        
        def _itemcf_recent_recall(self, user_id):
            """基于近期交互的ItemCF召回"""
            if user_id not in self.user_history:
                return {}
            
            user_items = self.user_history[user_id]
            recent_items = user_items[-self.recent_last_n:] if len(user_items) >= self.recent_last_n else user_items[:]
            
            candidate_items = {}
            for item in recent_items:
                if item in self.topk_similarities:
                    similar_items = self.topk_similarities[item]
                    for similar_item, similarity in similar_items.items():
                        if similar_item not in user_items:
                            candidate_items[similar_item] = candidate_items.get(similar_item, 0) + similarity
            
            return dict(sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)[:self.recall_num])
        
        def _hot_recall(self, user_id):
            """热门物品召回"""
            exclude_items = set(self.user_history.get(user_id, []))
            candidate = []
            count = 0
            for item in self.hot_items:
                if item not in exclude_items and count < self.hot_top_n:
                    candidate.append(item)
                    count += 1
            return candidate
        
        def _itemcf_long_term_recall(self, user_id):
            """基于长期兴趣的召回"""
            if user_id not in self.user_history:
                return {}
            
            user_interactions = self.user_history[user_id]
            if len(user_interactions) > self.long_term_threshold:
                long_term_items = user_interactions[:-self.long_term_threshold]
            else:
                long_term_items = user_interactions
            
            candidate_items = {}
            for item in long_term_items:
                if item in self.topk_similarities:
                    similar_items = self.topk_similarities[item]
                    for similar_item, similarity in similar_items.items():
                        if similar_item not in user_interactions:
                            candidate_items[similar_item] = candidate_items.get(similar_item, 0) + similarity
            
            return dict(sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)[:self.recall_num])
        
        def hybrid_recall(self, user_id):
            """混合召回策略"""
            recent_recall = self._itemcf_recent_recall(user_id)
            hot_recall = self._hot_recall(user_id)
            long_term_recall = self._itemcf_long_term_recall(user_id)
            
            ordered_candidates = (
                list(recent_recall.keys()) +
                hot_recall +
                list(long_term_recall.keys())
            )
            
            seen = set()
            final_recall = []
            for item in ordered_candidates:
                if item not in seen:
                    final_recall.append(item)
                    seen.add(item)
                if len(final_recall) >= self.recall_num:
                    break
            
            return final_recall
    
    # ==================== 6. 排序模型 ====================
    class RankingModel:
        """排序模型类，包含多种排序算法"""
        
        def __init__(self):
            self.lgb_model = None
            self.xgb_model = None
            
        def prepare_training_data(self, train_data, recall_system, movies, user_features):
            """准备排序训练数据"""
            data_all = train_data[['userId', 'movieId', 'rating']].copy()
            data_positive = data_all[data_all['rating'] >= 4]
            
            positive_movie = data_positive.groupby('userId')['movieId'].unique().apply(list).to_dict()
            negative_movie = {}
            
            for key, items in positive_movie.items():
                recall = recall_system.hybrid_recall(key)
                if not recall:
                    continue
                num = len(items) * 5 if len(items) <= 6 else 30
                sample_size = min(num, len(recall))
                negative_movie[key] = random.sample(recall, sample_size)
            
            neg_keys = []
            neg_list = []
            for key, item in negative_movie.items():
                neg_keys.extend(key for _ in range(len(item)))
                neg_list.extend(item)
            
            data_negative = pd.DataFrame({"userId": neg_keys, "movieId": neg_list})
            data_negative['rating'] = 0
            
            data_train = pd.concat([data_positive, data_negative])
            data_train = data_train.merge(movies, left_on="movieId", right_on="movieId", how="left")
            data_train = data_train.merge(user_features, left_on="userId", right_on="userId", how="left")
            data_train['current_year'] = 2025
            
            # 数据清洗
            data_train = data_train.dropna()
            numeric_cols = data_train.select_dtypes(include=[np.number]).columns
            data_train[numeric_cols] = data_train[numeric_cols].astype("int32")
            
            return data_train
        
        def train_lightgbm(self, data_train):
            time_start = time.time()
            """训练LightGBM排序模型"""
            X = data_train.drop(['userId', 'movieId', 'rating', 'title'], axis=1)
            y = data_train['rating']
            
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=data_train['userId']))
            
            train_df = data_train.iloc[train_idx].sort_values('userId')
            test_df = data_train.iloc[test_idx].sort_values('userId')
            
            X_train = train_df.drop(['userId', 'movieId', 'rating', 'title'], axis=1)
            X_test = test_df.drop(['userId', 'movieId', 'rating', 'title'], axis=1)
            y_train = train_df['rating']
            y_test = test_df['rating']
            
            train_groups = train_df['userId'].value_counts().sort_index().values
            test_groups = test_df['userId'].value_counts().sort_index().values
            
            train_data_lgb = lgb.Dataset(
                X_train, 
                label=y_train,
                group=train_groups,
                free_raw_data=False
            )
            
            test_data_lgb = lgb.Dataset(
                X_test, 
                label=y_test, 
                group=test_groups,
                reference=train_data_lgb,
                free_raw_data=False
            )
            
            # LightGBM参数调优
            lgb_params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'learning_rate': 0.1,
                'num_leaves': 55,
                'max_depth': -1,
                'min_data_in_leaf': 40,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbosity': -1
            }
            
            self.lgb_model = lgb.train(
                lgb_params,
                train_data_lgb,
                num_boost_round=1000,
                valid_sets=[test_data_lgb],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(50)
                ]
            )
            time_end = time.time()
            print(f"训练lgbm模型花费时间{time_end - time_start}")
            return self.lgb_model, X_test, y_test, test_groups
        
        def train_xgboost(self, data_train):
            """训练XGBoost排序模型"""
            time_start = time.time()
            X = data_train.drop(['userId', 'movieId', 'rating', 'title'], axis=1)
            y = data_train['rating']
            
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=data_train['userId']))
            
            train_df = data_train.iloc[train_idx].sort_values('userId')
            test_df = data_train.iloc[test_idx].sort_values('userId')
            
            X_train = train_df.drop(['userId', 'movieId', 'rating', 'title'], axis=1)
            X_test = test_df.drop(['userId', 'movieId', 'rating', 'title'], axis=1)
            y_train = train_df['rating']
            y_test = test_df['rating']
            
            train_groups = train_df['userId'].value_counts().sort_index().values
            test_groups = test_df['userId'].value_counts().sort_index().values
            
            # XGBoost数据准备
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # 设置group参数
            dtrain.set_group(train_groups)
            dtest.set_group(test_groups)
            
            # XGBoost参数调优
            xgb_params = {
                'objective': 'rank:pairwise',
                'eval_metric': 'ndcg@5',
                'learning_rate': 0.07,
                'max_depth': 5,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eta': 0.3,
                'seed': 42
            }
            
            self.xgb_model = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=500,
                evals=[(dtrain, 'train'), (dtest, 'eval')],
                early_stopping_rounds=50,
                verbose_eval=50
            )
            time_end = time.time()
            print(f"训练xgb模型花费时间{time_end - time_start}")
            return self.xgb_model, X_test, y_test, test_groups
        
        def evaluate_models(self, model_lgb, model_xgb, X_test, y_test, test_groups):
            """评估模型性能"""
            # LightGBM预测
            lgb_pred = model_lgb.predict(X_test)
            
            # XGBoost预测
            dtest = xgb.DMatrix(X_test)
            xgb_pred = model_xgb.predict(dtest)
            
            # 计算NDCG指标
            def calculate_ndcg(predictions, true_labels, groups, k=10):
                ndcg_scores = []
                start_idx = 0
                for group_size in groups:
                    end_idx = start_idx + group_size
                    group_pred = predictions[start_idx:end_idx]
                    group_true = true_labels[start_idx:end_idx]
                    if len(group_pred) > 0:
                        ndcg = ndcg_score([group_true], [group_pred], k=min(k, len(group_true)))
                        ndcg_scores.append(ndcg)
                    start_idx = end_idx
                return np.mean(ndcg_scores) if ndcg_scores else 0
            
            lgb_ndcg = calculate_ndcg(lgb_pred, y_test.values, test_groups, k=10)
            xgb_ndcg = calculate_ndcg(xgb_pred, y_test.values, test_groups, k=10)
            
            print(f"LightGBM NDCG@10: {lgb_ndcg:.4f}")
            print(f"XGBoost NDCG@10: {xgb_ndcg:.4f}")
            
            # 可视化对比
            models = ['LightGBM', 'XGBoost']
            ndcg_scores = [lgb_ndcg, xgb_ndcg]
            
            plt.figure(figsize=(8, 5))
            bars = plt.bar(models, ndcg_scores, color=['#3498db', '#e74c3c'])
            plt.title('模型性能对比 (NDCG@10)')
            plt.ylabel('NDCG Score')
            
            for bar, score in zip(bars, ndcg_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
            return lgb_ndcg, xgb_ndcg

# ==================== 7. 主程序 ====================
def main():
    """主程序"""
    print("开始电影推荐系统构建...")
    
    # 初始化推荐系统
    recommender = MovieRecommender()
    
    # 1. 加载数据
    try:
        rating, movies = recommender.load_data()
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 2. 探索性数据分析
    recommender.exploratory_analysis(rating, movies)
    
    # 3. 数据分割
    train_data, test_data = recommender.data_split(rating, movies)
    
    # 4. 特征工程
    movies, user_features = recommender.feature_engineering(train_data, movies)
    
    # 5. 计算相似度矩阵（召回准备）
    print("\n=== 计算相似度矩阵 ===")
    user_history = train_data.groupby('userId')['movieId'].unique().apply(list).to_dict()
    
    # 创建热门电影列表
    groups_movieId = train_data.groupby("movieId").agg(
        num=("userId", "nunique"),
        score_avg=("rating", "mean")
    ).reset_index().sort_values('num', ascending=False)
    hot_items = groups_movieId[(groups_movieId['num'] >= 100) & 
                               (groups_movieId['score_avg'] >= 3.0)]["movieId"].tolist()
    
    # 计算电影相似度
    """
    考虑直接用用户id和电影id作为索引，占用较大内存，这里转化为编码并一一对应
    """
    train_data['userId_cat'] = train_data['userId'].astype("category")
    train_data['movieId_cat'] = train_data['movieId'].astype("category")
    
    user_ids = train_data['userId_cat'].cat.codes
    movie_ids = train_data['movieId_cat'].cat.codes
    
    movie_code_to_id = dict(enumerate(train_data['movieId_cat'].cat.categories))
    
    user_movie_matrix = csr_matrix((train_data['rating'], (movie_ids, user_ids)))
    movie_similarity = cosine_similarity(user_movie_matrix, dense_output=False)
    
    # 获取Top10相似电影
    def get_top_similar_movies(similarity_matrix, movie_code_to_id, top_n=10):
        n_movies = similarity_matrix.shape[0]
        top_similar = {}
        for movie_i in range(n_movies):
            row = similarity_matrix.getrow(movie_i)
            similar_movie_codes = row.indices
            similarities = row.data
            mask = (similar_movie_codes != movie_i)
            similar_movie_codes = similar_movie_codes[mask]
            similarities = similarities[mask]
            
            if len(similarities) == 0:
                top_similar[movie_code_to_id[movie_i]] = {}
                continue
            
            sorted_indices = np.argsort(similarities)[::-1]
            top_indices = sorted_indices[:top_n]
            top_scores = similarities[top_indices]
            
            similar_dict = {
                movie_code_to_id[similar_movie_codes[i]]: float(score) 
                for i, score in zip(top_indices, top_scores)
            }
            top_similar[movie_code_to_id[movie_i]] = similar_dict
        
        return top_similar
    
    top10_similar = get_top_similar_movies(movie_similarity, movie_code_to_id, top_n=10)
    
    # 6. 构建召回系统
    print("\n=== 构建召回系统 ===")
    recall_system = recommender.RecallSystem(user_history, top10_similar, hot_items)
    
    # 测试召回
    test_user_id = 5
    recall_result = recall_system.hybrid_recall(test_user_id)
    print(f"用户{test_user_id}的召回结果（前10个）: {recall_result[:10]}")
    
    # 7. 构建排序模型
    print("\n=== 构建排序模型 ===")
    ranking_model = recommender.RankingModel()
    
    # 准备训练数据
    data_train_rank = ranking_model.prepare_training_data(train_data, recall_system, movies, user_features)
    data_train_rank['movieId'] = data_train_rank['movieId'].astype("int32")
    data_train_rank = data_train_rank.drop(columns = "genres")
    
    # 训练LightGBM模型
    print("\n训练LightGBM模型...")
    lgb_model, X_test_lgb, y_test_lgb, groups_lgb = ranking_model.train_lightgbm(data_train_rank)
    
    # 训练XGBoost模型
    print("\n训练XGBoost模型...")
    xgb_model, X_test_xgb, y_test_xgb, groups_xgb = ranking_model.train_xgboost(data_train_rank)
    
    # 评估模型
    print("\n=== 模型评估 ===")
    lgb_ndcg, xgb_ndcg = ranking_model.evaluate_models(lgb_model, xgb_model, X_test_lgb, y_test_lgb, groups_lgb)
    
    # 8. 测试推荐
    print("\n=== 测试推荐 ===")
    
    def get_recommendations(model_type, user_id, top_k=10):
        """获取推荐结果"""
        if model_type == 'lightgbm':
            recall_list = recall_system.hybrid_recall(user_id)
            if not recall_list:
                return []
            
            user_movie_feature = pd.DataFrame({"userId": [user_id] * len(recall_list), "movieId": recall_list})
            user_movie_feature = user_movie_feature.merge(movies, on="movieId", how="left")
            user_movie_feature = user_movie_feature.merge(user_features, on="userId", how="left")
            user_movie_feature['current_year'] = 2025
            
            feature_cols = user_movie_feature.drop(columns=["movieId", 'userId', "title", "genres"])
            scores = lgb_model.predict(feature_cols)
            movie_predict_dict = list(zip(recall_list, scores))
            ranked_movies = [movie for movie, score in sorted(movie_predict_dict, key=lambda x: x[1], reverse=True)]
            
            return ranked_movies[:top_k]
        
        elif model_type == 'xgboost':
            recall_list = recall_system.hybrid_recall(user_id)
            if not recall_list:
                return []
            
            user_movie_feature = pd.DataFrame({"userId": [user_id] * len(recall_list), "movieId": recall_list})
            user_movie_feature = user_movie_feature.merge(movies, on="movieId", how="left")
            user_movie_feature = user_movie_feature.merge(user_features, on="userId", how="left")
            user_movie_feature['current_year'] = 2025
            
            feature_cols = user_movie_feature.drop(columns=["movieId", 'userId', "title", "genres"])
            dmatrix = xgb.DMatrix(feature_cols)
            scores = xgb_model.predict(dmatrix)
            movie_predict_dict = list(zip(recall_list, scores))
            ranked_movies = [movie for movie, score in sorted(movie_predict_dict, key=lambda x: x[1], reverse=True)]
            
            return ranked_movies[:top_k]
    
    # 测试不同用户的推荐
    test_users = [1, 5, 10]
    for user_id in test_users:
        print(f"\n为用户 {user_id} 推荐:")
        lgb_rec = get_recommendations('lightgbm', user_id, 5)
        xgb_rec = get_recommendations('xgboost', user_id, 5)
        
        print(f"LightGBM推荐: {lgb_rec}")
        print(f"XGBoost推荐: {xgb_rec}")
        
        # 计算重合度
        overlap = len(set(lgb_rec) & set(xgb_rec))
        print(f"两个模型推荐重合度: {overlap}/5 ({overlap/5*100:.1f}%)")
    
    # 9. 最终评估
    print("\n=== 最终评估 (Recall@10) ===")
    
    def evaluate_recall(model_type, test_data, K=10):
        """计算Recall@K指标"""
        data_test_list = test_data.groupby("userId")['movieId'].apply(list).reset_index()
        tar_len = len(data_test_list)
        sum_ratio = 0
        cnt = 0
        
        for user_id in data_test_list['userId'].to_list():
            rec_movieId = get_recommendations(model_type, user_id, K)
            after_movieId = data_test_list[data_test_list['userId'] == user_id]['movieId'].iloc[0]
            hit_count = len(set(rec_movieId) & set(after_movieId))
            cnt += hit_count
            sum_ratio += hit_count / K if K else 0
        
        avg_ratio = sum_ratio / tar_len if tar_len else 0
        return avg_ratio, cnt
    
    lgb_recall, lgb_hits = evaluate_recall('lightgbm', test_data)
    xgb_recall, xgb_hits = evaluate_recall('xgboost', test_data)
    
    print(f"LightGBM Recall@10: {lgb_recall:.4f}, Total hits: {lgb_hits}")
    print(f"XGBoost Recall@10: {xgb_recall:.4f}, Total hits: {xgb_hits}")
    
    # 总结
    print("\n" + "="*50)
    print("项目总结:")
    print("="*50)
    print("1. 实现了基于ItemCF的混合召回系统")
    print("2. 实现了两种排序模型: LightGBM和XGBoost")
    print("3. LightGBM在NDCG@10上表现: {:.4f}".format(lgb_ndcg))
    print("4. XGBoost在NDCG@10上表现: {:.4f}".format(xgb_ndcg))
    print("5. LightGBM在Recall@10上表现: {:.4f}".format(lgb_recall))
    print("6. XGBoost在Recall@10上表现: {:.4f}".format(xgb_recall))
    
    return recommender, recall_system, ranking_model

# 运行主程序
if __name__ == "__main__":
    try:
        recommender, recall_system, ranking_model = main()
        print("\n电影推荐系统构建完成！")
    except Exception as e:
        print(f"程序运行出错: {e}")
