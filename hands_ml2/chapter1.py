import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors
import os
import matplotlib as mpl
import urllib.request


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    print(full_country_stats[["GDP per capita", "Life satisfaction"]].loc["United States"])
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indeices = list(set(range(36)) - set(remove_indices))

    sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indeices]
    remove_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]
    sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5, 3))
    plt.axis([0, 60000, 0, 10])
    # 指定字母在图上的显示位置
    position_text = {
        "Hungary": (5000, 1),
        "Korea": (18000, 1.7),
        "France": (29000, 2.4),
        "Australia": (40000, 3.0),
        "United States": (52000, 3.8),
    }
    for country, post_text in position_text.items():
        pos_data_x, pos_data_y = sample_data.loc[country]
        country = "U.S." if country == "United States" else country
        plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=post_text,
                     arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
        plt.plot(pos_data_x, pos_data_y, "ro")
    plt.xlabel("GDP per capita (USD)")
    plt.show()
    return sample_data


datapath = os.path.join("datasets", "lifesat", "")
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
# 下载数据
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
os.makedirs(datapath, exist_ok=True)
for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
    urllib.request.urlretrieve(url, datapath + filename)

# 加载数据
oecd_bli = pd.read_csv("datasets\\lifesat\\oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("datasets\\lifesat\\gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")
# 准备数据
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]
# 数据可视化，设置x轴为GDP，y轴为幸福指数
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()
# 选择一个线性回归
# model = sklearn.linear_model.LinearRegression()
# 选择使用K近邻算法预测数据
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
# 训练模型
model.fit(X, y)
# 根据输入的x轴的gdp数据预测该地的幸福指数
X_new = [[22587]]
print(model.predict(X_new))
