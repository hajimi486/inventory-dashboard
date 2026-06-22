# 朋友帮忙部署指南

## 你需要做的事（3步）

### 第1步：注册GitHub & 创建仓库

1. 打开 https://github.com 注册账号（或用已有账号登录）
2. 点右上角 `+` → `New repository`
3. 填写：
   - Repository name: `inventory-dashboard`
   - 选 **Public**（必须Public，Streamlit Cloud只支持公开仓库）
   - ✅ 勾选 `Add a README file`
4. 点 `Create repository`

### 第2步：上传代码文件

1. 让朋友下载部署包：https://www.coze.cn/s/FydHcWzNT8s/
2. 解压zip，得到以下文件结构：
   ```
   ├── .streamlit/
   │   └── config.toml
   ├── dashboard/
   │   ├── app.py
   │   └── data_update.py
   ├── data/
   │   └── raw/
   │       ├── 4materials_13weeks_forecast.csv
   │       ├── demand_forecast_error_2024.csv
   │       ├── inventory_params.csv
   │       ├── lead_time_distribution.csv
   │       ├── model_evaluation.csv
   │       ├── monte_carlo_results.csv
   │       ├── monthly_inventory_2024.csv
   │       ├── seal_ring_130weeks.csv
   │       └── seal_ring_static_dynamic_params.csv
   ├── results/
   │   ├── figures/（11个PNG图片）
   │   └── tables/（8个CSV文件）
   ├── requirements.txt
   └── README.md
   ```
3. 在仓库页面，逐个文件夹上传：
   - 点击 `Add file` → `Upload files`
   - 把解压出来的**所有文件**按目录结构拖进去
   - 注意：GitHub网页上传不支持空文件夹，但我们的文件夹里都有文件所以没问题
   - 每次上传后点 `Commit changes`
   - **.streamlit 文件夹**也要上传（里面是config.toml）

### 第3步：部署到Streamlit Cloud

1. 打开 https://share.streamlit.io/
2. 用GitHub账号登录
3. 点 `New app`
4. 填写：
   - Repository: 选择 `你的用户名/inventory-dashboard`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
5. 点 `Deploy!`
6. 等待2~5分钟，部署成功后会显示一个地址，格式类似：
   `https://inventory-dashboard-xxxx.streamlit.app`

**把这个地址发给我朋友（RootUser）就行！**

---

## ⚠️ 注意事项
- 仓库必须是 Public
- 上传时保持文件夹结构不变，特别是 `.streamlit/config.toml`
- 如果部署报错，检查 `requirements.txt` 是否在仓库根目录
- Streamlit Cloud免费版48小时无人访问会休眠，有人访问会自动唤醒（约30秒）
