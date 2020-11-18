import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving Figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# 下载housing.csv文件并解压在项目路径的datasets/housing目录下
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# 这个函数会返回一个包含所有数据的pandas DataFrame对象
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# 将数据集进行分割，测试集长度为len(data)*test_ratio
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[: test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# 返回小于指定hash值的数据集
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


class CombinedAttribute(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


if __name__ == '__main__':
    fetch_housing_data()
    housing = load_housing_data()
    print(housing.describe())
    # 每张图50个柱子，尺寸大小为20*15
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()
    # 方式1
    housing_with_id = housing.reset_index()  # 为housing数据集增加一个index列作为标签
    # 在此处，我们将数据集的80%用作训练集，20%用作测试集
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    # 方式2
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set2, test_set2 = split_train_test_by_id(housing_with_id, 0.2, "id")
    # 方式3 直接引用sklearn中的函数
    train_set3, test_set3 = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    housing["income_cat"].hist()
    plt.show()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        start_train_set = housing.loc[train_index]
        start_test_set = housing.loc[test_index]
    print(start_test_set["income_cat"].value_counts() / len(start_test_set))
    # 删除income_cat属性，将数据恢复原样
    for set_ in (start_train_set, start_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = start_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show()
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population", figsize=(10, 7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
    plt.legend()
    plt.show()
    images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
    os.makedirs(images_path, exist_ok=True)
    filename = "california.png"
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
    urllib.request.urlretrieve(url, os.path.join(images_path, filename))  # 将url上的图片文件从网络上下载并保存在本地
    california_img = mpimg.imread(os.path.join(images_path, filename))  # 读取下载到本地的加利福尼亚的地图
    ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10, 7),
                      s=housing['population'] / 100, label="Population",
                      c="median_house_value", cmap=plt.get_cmap("jet"),
                      colorbar=False, alpha=0.4)
    plt.imshow(california_img, extent=[-124.35, -113.80, 32.45, 42.05], alpha=0.5,
               cmap=plt.get_cmap("jet"))
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)

    prices = housing["median_house_value"]
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    cbar = plt.colorbar(ticks=tick_values / prices.max())
    cbar.ax.set_yticklabels(["$%dk" % (round(v / 1000)) for v in tick_values], fontsize=14)  # 为右侧的彩色柱状图添加数轴说明
    cbar.set_label('Median House Value', fontsize=16)
    plt.legend(fontsize=16)
    save_fig("california_housing_prices_plot")
    plt.show()
    corr_matrix = housing.corr()  # 显示各个列属性与房价中位数的相关性系数
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.show()
    # 尝试创建一些新属性，观察他们与房价中位数的相关性
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_room"] = housing["population"] / housing["households"]
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    # 回到一个干净的数据训练集
    housing = start_train_set.drop("median_house_value", axis=1)
    housing_labels = start_train_set["median_house_value"].copy()
    # 解决缺失的total_bedrooms部分属性值
    housing.dropna(subset=["total_bedrooms"])  # 放弃缺失该值的区域
    housing.drop("total_bedrooms", axis=1)  # 放弃整个属性
    median = housing["total_bedrooms"].median()  # 将缺失值的属性设置为该属性的中位数
    housing["total_bedrooms"].fillna(median, inplace=True)

    # 创建一个SimpleImputer实例，指定要用的属性的中位数值去替代该属性的缺失值
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)  # 创建一个没有ocean_proximity属性的数据副本
    imputer.fit(housing_num)  # 使用fit()方法将imputer实例适配到训练数据
    print(imputer.statistics_)
    print(housing_num.median().values)
    # 使用训练好的imputer将缺失值换成中位数值来完成训练集的转换，此处先fit再transform的做法可以使用fit_transform()函数替代
    X = imputer.transform(housing_num)
    # 将转换完的Numpy数组放回pandas DataFrame
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index=housing_num.index)
    housing_cat = housing[["ocean_proximity"]]
    print(housing_cat.head(10))
    # 使用OrdinalEncoder将分类特征转换为数值类型
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoder = ordinal_encoder.fit_transform(housing_cat)
    print(housing_cat_encoder[: 10])
    print(ordinal_encoder.categories_)
    # 此处对转换为数据类型的ocean_proximity采用读热编码
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot.toarray())

    attr_adder = CombinedAttribute(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)
    # 对数值属性流水线式的转换，即按照流水线的顺序对数值属性进行转换
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttribute()),
        ('std_scaler', StandardScaler()),  # 除了最后一个一个估算器外，其余都是转换器
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    print(housing_tr)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    # 获得数值列名称列表和类别列名称列表，之后构造一个ColumnTransformer
    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared.shape)
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    some_data = housing.iloc[: 5]
    some_labels = housing.iloc[: 5]
    some_data_prepared = full_pipeline.transform(some_data)
    print("Predictions:", lin_reg.predict(some_data_prepared))
    print("Labels:", list(some_labels))
    from sklearn.metrics import mean_squared_error
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)
    # 执行K折交叉验证
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    # 输出K折交叉验证的分数
    display_scores(tree_rmse_scores)

    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)

    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mes = mean_squared_error(housing_labels, housing_predictions)
    forest_rmes = np.sqrt(forest_mes)
    display_scores(forest_rmes)
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmes_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmes_scores)
    scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    print(pd.Series(np.sqrt(-scores)).describe())
    # 使用网格搜索，自动交叉验证来评估超参数值的所有可能
    from sklearn.model_selection import GridSearchCV
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring="neg_mean_squared_error",
                               return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)
    # 得到最佳的参数组合
    print(grid_search.best_params_)
    # 得到最好的估算器
    print(grid_search.best_estimator_)
    # 获得评估分数
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    # 使用随即搜索，为每个超参数值选择一个随机值
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring="neg_mean_squared_error",
                                    random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # 分析最佳模型及其误差
    feature_importances = grid_search.best_estimator_.feature_importances_
    print(feature_importances)
    extra_attribs = ["rooms_per_hhold", "pop_per_hhlod", "bedrooms_per_room"]
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs = num_attribs + extra_attribs + cat_one_hot_attribs
    sorted(zip(feature_importances, attributes), reverse=True)

    final_model = grid_search.best_estimator_
    X_test = start_test_set.drop("median_house_value", axis=1)
    y_test = start_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("rmse is:", final_rmse)
