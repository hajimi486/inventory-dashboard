"""
Streamlit 库存管理决策系统 —— 面向汽车零部件的库存管理优化
运行方式: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Data update page module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_update import render as render_data_update

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULT_TABLE_DIR = os.path.join(BASE_DIR, "results", "tables")
RESULT_FIG_DIR = os.path.join(BASE_DIR, "results", "figures")

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="库存管理决策系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS - 学术风格
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 0.8rem; color: white;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; }
    .saving-positive { color: #27ae60; font-weight: 700; }
    .saving-negative { color: #e74c3c; font-weight: 700; }
    .section-header { 
        font-size: 1.3rem; font-weight: 600; color: #2c3e50;
        border-bottom: 2px solid #3498db; padding-bottom: 0.3rem;
        margin-top: 1rem;
    }
    .stSidebar { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 正确的物料主数据（与报告/inventory_params.csv一致）
# ============================================================
MATERIALS_MASTER = {
    '碟形弹簧': {
        'category': 'Ⅰ-1', 'category_name': '战略型（平稳）',
        'strategy': '(R,Q)', 'strategy_full': '(R,Q) 连续检查策略',
        'cv': 0.06, 'cv_type': '平稳',
        'd_bar': 13622, 'sigma_d': 785, 'price': 10.57,
        'holding': 1.40, 'stockout': 154, 'order_cost': 18650,
        'SL': 0.99, 'leadtime_weeks': 1,
        'SS': 1830, 'R': 15452, 'Q': 19051, 'S': None, 'T_days': None,
    },
    '止推环': {
        'category': 'Ⅱ-1', 'category_name': '瓶颈型（平稳）',
        'strategy': '(R,S)', 'strategy_full': '(R,S) 连续检查策略',
        'cv': 0.10, 'cv_type': '平稳',
        'd_bar': 2055, 'sigma_d': 208, 'price': 20.50,
        'holding': 1.40, 'stockout': 127, 'order_cost': 6500,
        'SL': 0.95, 'leadtime_weeks': 2,
        'SS': 486, 'R': 4596, 'Q': None, 'S': 8706, 'T_days': None,
    },
    '定位轴套': {
        'category': 'Ⅳ-1', 'category_name': '一般型（平稳）',
        'strategy': '(T,S)', 'strategy_full': '(T,S) 周期检查策略',
        'cv': 0.26, 'cv_type': '平稳',
        'd_bar': 884, 'sigma_d': 230, 'price': 7.38,
        'holding': 1.05, 'stockout': 60, 'order_cost': 4100,
        'SL': 0.95, 'leadtime_weeks': 2,
        'SS': 849, 'R': None, 'Q': None, 'S': 5269, 'T_days': 21,
    },
    '密封圈': {
        'category': 'Ⅲ-2', 'category_name': '杠杆型（非平稳）',
        'strategy': '静动结合', 'strategy_full': '静动结合策略',
        'cv': 0.46, 'cv_type': '非平稳',
        'd_bar': 1400, 'sigma_d': 560, 'price': 15.50,
        'holding': 1.75, 'stockout': 115, 'order_cost': 12000,
        'SL': 0.95, 'leadtime_weeks': 1,
        'SS': 500, 'R': None, 'Q': None, 'S': None, 'T_days': None,
    },
}

# FCM 4大类定义
FCM_CATEGORIES = {
    'Ⅰ-战略型': {'count': 319, 'desc': '高价值+高关键性，缺货影响大', 'representative': '碟形弹簧'},
    'Ⅱ-瓶颈型': {'count': 65, 'desc': '高供应风险，替代性低', 'representative': '止推环'},
    'Ⅲ-杠杆型': {'count': 193, 'desc': '高消耗量，替代品较多', 'representative': '密封圈'},
    'Ⅳ-一般型': {'count': 76, 'desc': '低价值+低风险', 'representative': '定位轴套'},
}

# ============================================================
# 数据加载（带缓存）
# ============================================================
@st.cache_data
def load_data():
    data = {}
    
    # 原始数据
    seal_path = os.path.join(DATA_DIR, "seal_ring_130weeks.csv")
    if os.path.exists(seal_path):
        data['seal_ring'] = pd.read_csv(seal_path)
    
    forecast_path = os.path.join(DATA_DIR, "4materials_13weeks_forecast.csv")
    if os.path.exists(forecast_path):
        data['forecast_13w'] = pd.read_csv(forecast_path)
    
    params_path = os.path.join(DATA_DIR, "inventory_params.csv")
    if os.path.exists(params_path):
        data['inventory_params'] = pd.read_csv(params_path)
    
    monthly_path = os.path.join(DATA_DIR, "monthly_inventory_2024.csv")
    if os.path.exists(monthly_path):
        data['monthly_inv'] = pd.read_csv(monthly_path)
    
    demand_err_path = os.path.join(DATA_DIR, "demand_forecast_error_2024.csv")
    if os.path.exists(demand_err_path):
        data['demand_error'] = pd.read_csv(demand_err_path)
    
    # 结果数据
    model_cmp_path = os.path.join(RESULT_TABLE_DIR, "model_comparison.csv")
    if os.path.exists(model_cmp_path):
        data['model_comparison'] = pd.read_csv(model_cmp_path)
    
    mc_cmp_path = os.path.join(RESULT_TABLE_DIR, "monte_carlo_comparison.csv")
    if os.path.exists(mc_cmp_path):
        data['mc_comparison'] = pd.read_csv(mc_cmp_path)
    
    strategy_path = os.path.join(RESULT_TABLE_DIR, "inventory_strategy_params.csv")
    if os.path.exists(strategy_path):
        data['strategy_params'] = pd.read_csv(strategy_path)
    
    arima_metrics_path = os.path.join(RESULT_TABLE_DIR, "arima_metrics.csv")
    if os.path.exists(arima_metrics_path):
        data['arima_metrics'] = pd.read_csv(arima_metrics_path)
    
    arima_fc_path = os.path.join(RESULT_TABLE_DIR, "arima_forecast_compare.csv")
    if os.path.exists(arima_fc_path):
        data['arima_forecast'] = pd.read_csv(arima_fc_path)
    
    model_fc_path = os.path.join(RESULT_TABLE_DIR, "model_forecast_compare.csv")
    if os.path.exists(model_fc_path):
        data['model_forecast'] = pd.read_csv(model_fc_path)
    
    # 4物料预测评估数据（与报告表4一致）
    model_eval_path = os.path.join(DATA_DIR, "model_evaluation.csv")
    if os.path.exists(model_eval_path):
        data['model_evaluation'] = pd.read_csv(model_eval_path)
    
    # 结构性指标（周转率/平均库存/资金占用/缺货占比）
    structural_path = os.path.join(RESULT_TABLE_DIR, "structural_metrics.csv")
    if os.path.exists(structural_path):
        data['structural_metrics'] = pd.read_csv(structural_path)
    
    return data

data = load_data()

# ============================================================
# 侧边栏
# ============================================================
st.sidebar.markdown("## 📊 库存管理决策系统")
st.sidebar.markdown("---")
st.sidebar.markdown("**面向汽车零部件的库存管理优化**")
st.sidebar.markdown("基于FCM聚类 + 多模型预测 + 蒙特卡洛仿真")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航",
    ["🏠 系统总览", "📈 需求预测分析", "🏷️ FCM聚类分类", "📦 库存策略优化", "🎲 蒙特卡洛仿真", "📋 策略参数详情", "🔄 数据更新"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**项目信息**
- 2026年机械工程创新创意大赛
- 赛项九：工业工程与精益管理创新
- 主题：数驱精益，智融创新
""")

# ============================================================
# 工具函数
# ============================================================
def format_currency(val):
    """格式化货币"""
    if abs(val) >= 10000:
        return f"{val/10000:.1f}万元"
    return f"{val:,.0f}元"

WAN = 10000  # 万元换算因子

def calc_saving_class(saving_rate):
    """节约率颜色类"""
    return "saving-positive" if saving_rate > 0 else "saving-negative"

# ============================================================
# 页面1: 系统总览
# ============================================================
if page == "🏠 系统总览":
    st.markdown('<p class="main-title">📊 库存管理决策系统</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 顶部关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    if 'mc_comparison' in data:
        mc = data['mc_comparison']
        # 使用需求采购口径
        h_col = 'h_total_demand' if 'h_total_demand' in mc.columns else 'h_company_cost'
        opt_col = 'opt_total_demand' if 'opt_total_demand' in mc.columns else 'optimized_cost'
        total_h = mc[h_col].sum()
        total_opt = mc[opt_col].sum()
        total_saving = total_h - total_opt
        avg_rate = (total_saving / total_h * 100) if total_h > 0 else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{format_currency(total_h)}</div>
                <div class="metric-label">H公司原季度总成本</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{format_currency(total_opt)}</div>
                <div class="metric-label">优化后季度总成本</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);">
                <div class="metric-value">{format_currency(total_saving)}</div>
                <div class="metric-label">季度节约总额</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);">
                <div class="metric-value">{avg_rate:.1f}%</div>
                <div class="metric-label">综合节约率</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 4种材料分项对比
    col_left, col_right = st.columns(2)
    
    if 'mc_comparison' in data:
        mc = data['mc_comparison']
        
        with col_left:
            st.markdown('<p class="section-header">各材料成本对比（需求驱动采购口径）</p>', unsafe_allow_html=True)
            
            material_col = 'material' if 'material' in mc.columns else mc.columns[0]
            h_col = 'h_total_demand' if 'h_total_demand' in mc.columns else 'h_company_cost'
            opt_col = 'opt_total_demand' if 'opt_total_demand' in mc.columns else 'optimized_cost'
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='H公司原策略',
                x=mc[material_col],
                y=mc[h_col] / WAN,
                marker_color='#e74c3c',
                marker_opacity=0.8
            ))
            fig.add_trace(go.Bar(
                name='优化策略',
                x=mc[material_col],
                y=mc[opt_col] / WAN,
                marker_color='#27ae60',
                marker_opacity=0.8
            ))
            fig.update_layout(
                barmode='group',
                yaxis_title='季度总成本（万元）',
                xaxis_title='材料',
                height=400,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown('<p class="section-header">各材料节约率</p>', unsafe_allow_html=True)
            
            saving_rates = []
            for _, row in mc.iterrows():
                if row[h_col] > 0:
                    saving_rates.append((row[h_col] - row[opt_col]) / row[h_col] * 100)
                else:
                    saving_rates.append(0)
            
            colors = ['#27ae60' if r > 0 else '#e74c3c' for r in saving_rates]
            
            fig2 = go.Figure(go.Bar(
                x=mc[material_col],
                y=saving_rates,
                marker_color=colors,
                text=[f"{r:.1f}%" for r in saving_rates],
                textposition='outside'
            ))
            fig2.update_layout(
                yaxis_title='节约率（%）',
                xaxis_title='材料',
                height=400,
                template='plotly_white',
                yaxis_ticksuffix='%'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # 模型预测对比
    if 'model_evaluation' in data:
        st.markdown('<p class="section-header">预测模型性能概览</p>', unsafe_allow_html=True)
        eval_df = data['model_evaluation']
        compare_df = eval_df[eval_df['model'].isin(['ARIMA', 'ARIMA-LSTM'])][['material', 'model', 'rmse', 'r2']].copy()
        compare_df.columns = ['物料', '模型', 'RMSE', 'R²']
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
    
    # FCM分类概览
    st.markdown('<p class="section-header">FCM聚类分类概览</p>', unsafe_allow_html=True)
    fcm_overview = pd.DataFrame([
        {'类别': k, '种类数': v['count'], '特征描述': v['desc'], '代表物料': v['representative']}
        for k, v in FCM_CATEGORIES.items()
    ])
    st.dataframe(fcm_overview, use_container_width=True, hide_index=True)

# ============================================================
# 页面2: 需求预测分析
# ============================================================
elif page == "📈 需求预测分析":
    st.markdown('<p class="main-title">📈 需求预测分析</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # H公司预测偏差（表2数据）
    if 'demand_error' in data:
        st.markdown('<p class="section-header">H公司2024年月度需求预测偏差</p>', unsafe_allow_html=True)
        de = data['demand_error']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=de['month'].astype(str) + '月',
            y=de['relative_error'] * 100,
            marker_color=['#e74c3c' if abs(e) > 0.15 else '#3498db' for e in de['relative_error']],
            text=[f"{e*100:.1f}%" for e in de['relative_error']],
            textposition='outside'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title='H公司经验判断法月度预测偏差（MAPE=18.64%）',
            yaxis_title='相对误差（%）',
            xaxis_title='月份',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(de, use_container_width=True, hide_index=True)
    
    # ARIMA分析
    if 'seal_ring' in data:
        st.markdown('<p class="section-header">密封圈130周需求序列</p>', unsafe_allow_html=True)
        
        seal = data['seal_ring']
        num_cols = seal.select_dtypes(include=[np.number]).columns.tolist()
        week_col = 'week' if 'week' in seal.columns else None
        y_col = 'demand' if 'demand' in seal.columns else (num_cols[-1] if num_cols else None)
        
        if y_col:
            fig = px.line(seal, x=week_col, y=y_col,
                         title='密封圈历史需求序列（130周）')
            fig.update_layout(template='plotly_white', height=400, xaxis_title='周', yaxis_title='需求量')
            st.plotly_chart(fig, use_container_width=True)
    
    # 多模型预测对比 - 使用model_evaluation.csv（与报告表4一致）
    if 'model_evaluation' in data:
        st.markdown('<p class="section-header">4种物料预测模型性能对比</p>', unsafe_allow_html=True)
        
        eval_df = data['model_evaluation']
        
        # 只展示ARIMA和ARIMA-LSTM（不含H公司）
        compare_df = eval_df[eval_df['model'].isin(['ARIMA', 'ARIMA-LSTM'])].copy()
        
        # 画分组柱状图
        materials_list = compare_df['material'].unique()
        metrics = ['rmse', 'r2']
        metric_labels = {'rmse': 'RMSE', 'r2': 'R²'}
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=['RMSE', 'R²'])
        
        for model_name in ['ARIMA', 'ARIMA-LSTM']:
            sub = compare_df[compare_df['model'] == model_name]
            color = '#e74c3c' if model_name == 'ARIMA' else '#27ae60'
            fig.add_trace(go.Bar(x=sub['material'], y=sub['rmse'], name=model_name,
                                marker_color=color, showlegend=True), row=1, col=1)
            fig.add_trace(go.Bar(x=sub['material'], y=sub['r2'], name=model_name,
                                marker_color=color, showlegend=False), row=1, col=2)
        
        fig.update_layout(height=400, template='plotly_white',
                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_yaxes(title_text='RMSE', row=1, col=1)
        fig.update_yaxes(title_text='R²', row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)
        
        # 展示完整数据表
        display_eval = compare_df[['material', 'model', 'rmse', 'mae', 'r2']].copy()
        display_eval.columns = ['物料', '模型', 'RMSE', 'MAE', 'R²']
        st.dataframe(display_eval, use_container_width=True, hide_index=True)
        
        # RMSE降低率汇总
        st.markdown('<p class="section-header">ARIMA-LSTM相对ARIMA的改善</p>', unsafe_allow_html=True)
        improvements = []
        for mat in materials_list:
            arima_row = compare_df[(compare_df['material'] == mat) & (compare_df['model'] == 'ARIMA')]
            lstm_row = compare_df[(compare_df['material'] == mat) & (compare_df['model'] == 'ARIMA-LSTM')]
            if len(arima_row) > 0 and len(lstm_row) > 0:
                rmse_drop = (arima_row['rmse'].values[0] - lstm_row['rmse'].values[0]) / arima_row['rmse'].values[0] * 100
                r2_gain = lstm_row['r2'].values[0] - arima_row['r2'].values[0]
                improvements.append({
                    '物料': mat,
                    'RMSE降低': f"{rmse_drop:.1f}%",
                    'R²提升': f"+{r2_gain:.2f}",
                })
        if improvements:
            st.dataframe(pd.DataFrame(improvements), use_container_width=True, hide_index=True)
    
    # 13周预测值
    if 'model_forecast' in data:
        st.markdown('<p class="section-header">密封圈13周预测值对比</p>', unsafe_allow_html=True)
        fc_df = data['model_forecast']
        st.dataframe(fc_df, use_container_width=True, hide_index=True)
    
    # ARIMA预测详情
    if 'arima_forecast' in data:
        st.markdown('<p class="section-header">ARIMA密封圈预测详情</p>', unsafe_allow_html=True)
        af = data['arima_forecast']
        
        actual_col = 'actual'
        forecast_col = 'arima_forecast'
        x_col = 'week' if 'week' in af.columns else None
        
        if actual_col in af.columns and forecast_col in af.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=af[x_col] if x_col else af.index, 
                                    y=af[actual_col], mode='lines+markers', 
                                    name='实际值', line=dict(color='#2c3e50')))
            fig.add_trace(go.Scatter(x=af[x_col] if x_col else af.index, 
                                    y=af[forecast_col], mode='lines+markers', 
                                    name='ARIMA预测', line=dict(color='#e74c3c', dash='dash')))
            fig.update_layout(title='密封圈ARIMA(3,2,3)预测 vs 实际',
                            template='plotly_white', height=400,
                            xaxis_title='周', yaxis_title='需求量')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 页面3: FCM聚类分类
# ============================================================
elif page == "🏷️ FCM聚类分类":
    st.markdown('<p class="main-title">🏷️ FCM模糊聚类分类</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 聚类结果展示
    st.markdown('<p class="section-header">FCM聚类结果（4大类）</p>', unsafe_allow_html=True)
    
    cluster_data = {
        '类别': ['Ⅰ-战略型', 'Ⅱ-瓶颈型', 'Ⅲ-杠杆型', 'Ⅳ-一般型'],
        '种类数': [319, 65, 193, 76],
        '特征描述': ['高价值+高关键性，缺货影响大', '高供应风险，替代性低', '高消耗量，替代品较多', '低价值+低风险'],
        '代表物料': ['碟形弹簧', '止推环', '密封圈', '定位轴套'],
    }
    cluster_df = pd.DataFrame(cluster_data)
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)
    
    # 需求变异系数细分
    st.markdown('<p class="section-header">综合分类结果（FCM + 需求变异系数CV）</p>', unsafe_allow_html=True)
    
    subcat_data = {
        '类别': ['Ⅰ-战略型', 'Ⅱ-瓶颈型', 'Ⅲ-杠杆型', 'Ⅳ-一般型'],
        '平稳(CV≤0.4)': ['Ⅰ-1(247种)', 'Ⅱ-1(45种)', 'Ⅲ-1(150种)', 'Ⅳ-1(55种)'],
        '非平稳(CV>0.4)': ['Ⅰ-2(72种)', 'Ⅱ-2(20种)', 'Ⅲ-2(43种)', 'Ⅳ-2(21种)'],
    }
    subcat_df = pd.DataFrame(subcat_data)
    st.dataframe(subcat_df, use_container_width=True, hide_index=True)
    
    # 4种核心物料分类详情
    st.markdown('<p class="section-header">4种核心物料细分结果</p>', unsafe_allow_html=True)
    
    core_class = pd.DataFrame({
        '物料': ['碟形弹簧', '止推环', '定位轴套', '密封圈'],
        '分类': ['Ⅰ-1 战略型（平稳）', 'Ⅱ-1 瓶颈型（平稳）', 'Ⅳ-1 一般型（平稳）', 'Ⅲ-2 杠杆型（非平稳）'],
        '变异系数CV': [0.06, 0.10, 0.26, 0.46],
        '推荐策略': ['(R,Q) 连续检查', '(R,S) 连续检查', '(T,S) 周期检查', '静动结合'],
    })
    st.dataframe(core_class, use_container_width=True, hide_index=True)
    
    # 雷达图 - 3维（CV, 需求均值, 波动性）
    st.markdown('<p class="section-header">4种原材料需求特征雷达图</p>', unsafe_allow_html=True)
    
    # 用plotly画正确的3维雷达图
    categories = ['变异系数CV', '需求均值', '波动系数(σ/√d̄)']
    mat_names = ['碟形弹簧', '止推环', '定位轴套', '密封圈']
    colors_r = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    # 原始值
    raw_vals = {
        '变异系数CV': [0.06, 0.10, 0.26, 0.46],
        '需求均值': [13622, 2055, 884, 1400],
        '波动系数(σ/√d̄)': [785/np.sqrt(13622), 208/np.sqrt(2055), 230/np.sqrt(884), 560/np.sqrt(1400)],
    }
    
    # 归一化到0-1
    arr = np.array([[raw_vals[c][i] for c in categories] for i in range(4)])
    arr_min = arr.min(axis=0)
    arr_max = arr.max(axis=0)
    arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-10)
    
    fig = go.Figure()
    for i, mat in enumerate(mat_names):
        fig.add_trace(go.Scatterpolar(
            r=arr_norm[i].tolist() + [arr_norm[i][0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=mat,
            opacity=0.5,
            line=dict(color=colors_r[i], width=2)
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0])),
        showlegend=True, height=500, template='plotly_white',
        title='4种原材料需求特征雷达图（归一化）'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 原始值对照表
    st.markdown('<p class="section-header">特征原始值</p>', unsafe_allow_html=True)
    feature_raw = pd.DataFrame({
        '物料': mat_names,
        '变异系数CV': [0.06, 0.10, 0.26, 0.46],
        '周均需求(件)': [13622, 2055, 884, 1400],
        '需求标准差(件/周)': [785, 208, 230, 560],
        '波动系数(σ/√d̄)': [f"{v:.2f}" for v in [785/np.sqrt(13622), 208/np.sqrt(2055), 230/np.sqrt(884), 560/np.sqrt(1400)]],
        '需求特性': ['平稳', '平稳', '平稳', '非平稳'],
    })
    st.dataframe(feature_raw, use_container_width=True, hide_index=True)
    
    # 聚类散点图
    st.markdown('<p class="section-header">聚类散点图</p>', unsafe_allow_html=True)
    scatter_path = os.path.join(RESULT_FIG_DIR, "fcm_clustering.png")
    if os.path.exists(scatter_path):
        st.image(scatter_path, use_container_width=True)

# ============================================================
# 页面4: 库存策略优化
# ============================================================
elif page == "📦 库存策略优化":
    st.markdown('<p class="main-title">📦 库存策略优化</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 策略映射
    st.markdown('<p class="section-header">FCM聚类 → 分类-策略映射</p>', unsafe_allow_html=True)
    
    strategy_map = pd.DataFrame({
        '子类': ['Ⅰ-1 战略型（平稳）', 'Ⅱ-1 瓶颈型（平稳）', 'Ⅳ-1 一般型（平稳）', 'Ⅲ-2 杠杆型（非平稳）'],
        '需求特征': ['高价值+平稳需求', '中价值+平稳需求', '低价值+平稳需求', '中价值+非平稳需求'],
        '推荐策略': ['(R,Q) 连续检查', '(R,S) 连续检查', '(T,S) 周期检查', '静动结合策略'],
        '选择理由': [
            '实时监控，经济批量订货，最高服务水平',
            '触发即补至目标水平，简化补货决策',
            '减少监控频率，定期补充，降低管理成本',
            '静态基准量+动态调整量，适应需求波动'
        ],
        '适用物料': ['碟形弹簧', '止推环', '定位轴套', '密封圈'],
    })
    st.dataframe(strategy_map, use_container_width=True, hide_index=True)
    
    # 策略参数（从inventory_params.csv读取）
    st.markdown('<p class="section-header">各材料库存策略参数</p>', unsafe_allow_html=True)
    
    if 'strategy_params' in data:
        st.dataframe(data['strategy_params'], use_container_width=True, hide_index=True)
    else:
        params_data = pd.DataFrame({
            '材料': ['碟形弹簧', '止推环', '定位轴套', '密封圈'],
            '策略': ['(R,Q)', '(R,S)', '(T,S)', '静动结合'],
            '安全库存SS': [1830, 486, 849, 500],
            '订货点R': [15452, 4596, '—', '动态调整'],
            '订货量Q/最高库存S': [19051, 8706, 5269, '动态调整'],
            '检查周期T': ['—', '—', '21天', '—'],
            '服务水平': ['99%', '95%', '95%', '95%'],
        })
        st.dataframe(params_data, use_container_width=True, hide_index=True)
    
    # 完整参数对比
    st.markdown('<p class="section-header">4种物料关键参数一览</p>', unsafe_allow_html=True)
    
    full_params = pd.DataFrame({
        '物料': ['碟形弹簧', '止推环', '定位轴套', '密封圈'],
        '分类': ['Ⅰ-1 战略型（平稳）', 'Ⅱ-1 瓶颈型（平稳）', 'Ⅳ-1 一般型（平稳）', 'Ⅲ-2 杠杆型（非平稳）'],
        '策略': ['(R,Q)', '(R,S)', '(T,S)', '静动结合'],
        '周均需求': ['13,622件', '2,055件', '884件', '1,400件'],
        '变异系数CV': [0.06, 0.10, 0.26, 0.46],
        '单价(元)': [10.57, 20.50, 7.38, 15.50],
        '服务水平': ['99%', '95%', '95%', '95%'],
        '核心参数': ['R=15,452, Q=19,051', 'R=4,596, S=8,706', 'T=21天, S=5,269', '周期13周'],
    })
    st.dataframe(full_params, use_container_width=True, hide_index=True)
    
    # 月度库存趋势
    if 'monthly_inv' in data:
        st.markdown('<p class="section-header">H公司2024年月度库存占用</p>', unsafe_allow_html=True)
        mi = data['monthly_inv']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=mi['month'].astype(str)+'月', y=mi['raw_material_KEUR'], 
                            name='原材料占用', marker_color='#3498db'))
        fig.add_trace(go.Scatter(x=mi['month'].astype(str)+'月', y=mi['raw_material_pct']*100,
                                mode='lines+markers', name='原材料占比(%)', 
                                yaxis='y2', line=dict(color='#e74c3c')))
        fig.add_shape(type="line", x0=0, x1=1, y0=70, y1=70,
                      xref="paper", yref="y2",
                      line=dict(dash="dash", color="gray", width=1))
        fig.add_annotation(x=0.98, y=70, text="行业理想值70%", yref="y2",
                          showarrow=False, font=dict(size=10, color="gray"),
                          xref="paper", xanchor="right")
        fig.update_layout(
            title='H公司2024年月度库存资金占用',
            yaxis_title='原材料占用(KEUR)',
            yaxis2=dict(title='原材料占比(%)', overlaying='y', side='right', range=[60, 90]),
            height=450, template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 页面5: 蒙特卡洛仿真
# ============================================================
elif page == "🎲 蒙特卡洛仿真":
    st.markdown('<p class="main-title">🎲 蒙特卡洛仿真结果</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    **仿真设置**: 5000次仿真 × 91天，需求驱动采购成本口径
    
    > 采购成本按需求驱动口径计算（demand × unit_price），消除lost sales下缺货少买的虚假节约效应。
    > 两种策略面对相同需求，采购成本相同，差异仅来自运营效率（持有+订货+缺货）。
    """)
    
    if 'mc_comparison' in data:
        mc = data['mc_comparison']
        material_col = mc.columns[0]
        
        # 成本分项对比
        st.markdown('<p class="section-header">成本分项对比（堆叠柱状图）</p>', unsafe_allow_html=True)
        
        has_breakdown = all(c in mc.columns for c in ['h_holding', 'h_ordering', 'h_stockout'])
        
        if has_breakdown:
            materials = mc[material_col].tolist()
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=['H公司原策略', '优化策略'])
            
            fig.add_trace(go.Bar(x=materials, y=mc['h_holding'] / WAN, name='持有成本', 
                                marker_color='#3498db'), row=1, col=1)
            fig.add_trace(go.Bar(x=materials, y=mc['h_ordering'] / WAN, name='订货成本', 
                                marker_color='#f39c12'), row=1, col=1)
            fig.add_trace(go.Bar(x=materials, y=mc['h_stockout'] / WAN, name='缺货成本', 
                                marker_color='#e74c3c'), row=1, col=1)
            
            opt_hold_col = 'opt_holding' if 'opt_holding' in mc.columns else mc.columns[4]
            opt_ord_col = 'opt_ordering' if 'opt_ordering' in mc.columns else mc.columns[5]
            opt_so_col = 'opt_stockout' if 'opt_stockout' in mc.columns else mc.columns[6]
            
            fig.add_trace(go.Bar(x=materials, y=mc[opt_hold_col] / WAN, name='持有成本', 
                                marker_color='#3498db', showlegend=False), row=1, col=2)
            fig.add_trace(go.Bar(x=materials, y=mc[opt_ord_col] / WAN, name='订货成本', 
                                marker_color='#f39c12', showlegend=False), row=1, col=2)
            fig.add_trace(go.Bar(x=materials, y=mc[opt_so_col] / WAN, name='缺货成本', 
                                marker_color='#e74c3c', showlegend=False), row=1, col=2)
            
            fig.update_layout(barmode='stack', height=450, template='plotly_white',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
            fig.update_yaxes(title_text='成本（万元）', row=1, col=1)
            fig.update_yaxes(title_text='成本（万元）', row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)
        
        # 详细数据表
        st.markdown('<p class="section-header">仿真数据明细</p>', unsafe_allow_html=True)
        st.dataframe(mc, use_container_width=True, hide_index=True)
        
        # 节约率汇总
        st.markdown('<p class="section-header">各材料节约率（需求驱动采购口径）</p>', unsafe_allow_html=True)
        
        h_col_mc = 'h_total_demand' if 'h_total_demand' in mc.columns else 'h_company_cost'
        opt_col_mc = 'opt_total_demand' if 'opt_total_demand' in mc.columns else 'optimized_cost'
        
        for _, row in mc.iterrows():
            mat = row[material_col]
            h_cost = row[h_col_mc]
            o_cost = row[opt_col_mc]
            saving = h_cost - o_cost
            rate = saving / h_cost * 100 if h_cost > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("材料", mat)
            col2.metric("H公司成本", format_currency(h_cost))
            col3.metric("优化后成本", format_currency(o_cost))
            col4.metric("节约率", f"{rate:.1f}%", delta=format_currency(saving))
    
    # 结构性指标：周转率、平均库存、资金占用、缺货占比
    if 'structural_metrics' in data:
        sm = data['structural_metrics']
        mat_col_sm = sm.columns[0]
        
        st.markdown("---")
        st.markdown('<p class="section-header">库存结构性指标对比</p>', unsafe_allow_html=True)
        st.markdown("""
        > 优化策略的核心机制是"以持有换缺货"：通过合理提高安全库存水平，持有成本适度上升，
        > 但缺货成本大幅下降，运营总成本显著降低。周转率下降和资金占用上升是换取更高服务水平和更低总成本的合理代价。
        """)
        
        # 顶部汇总指标卡
        total_holding_h = mc['h_holding'].sum()
        total_holding_opt = mc['opt_holding'].sum()
        total_stockout_h = mc['h_stockout'].sum()
        total_stockout_opt = mc['opt_stockout'].sum()
        holding_change = (total_holding_opt - total_holding_h) / total_holding_h * 100
        stockout_change = (total_stockout_opt - total_stockout_h) / total_stockout_h * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("持有成本变化", f"+{holding_change:.1f}%", delta="以持有换缺货")
        col2.metric("缺货成本变化", f"{stockout_change:.1f}%", delta="大幅下降", delta_color="normal")
        col3.metric("库存总成本节约", "9.9%", delta="41.0万元/季")
        col4.metric("缺货成本占比", f"{total_stockout_h/(mc['h_operational'].sum())*100:.1f}% → {total_stockout_opt/(mc['opt_operational'].sum())*100:.1f}%", delta="显著改善")
        
        # 周转率对比柱状图
        st.markdown('<p class="section-header">季度周转率对比</p>', unsafe_allow_html=True)
        
        fig_to = go.Figure()
        materials_sm = sm[mat_col_sm].tolist()
        fig_to.add_trace(go.Bar(
            x=materials_sm, y=sm['turnover_h_quarterly'],
            name='H公司原策略', marker_color='#3498db', width=0.35
        ))
        fig_to.add_trace(go.Bar(
            x=materials_sm, y=sm['turnover_opt_quarterly'],
            name='优化策略', marker_color='#e67e22', width=0.35
        ))
        fig_to.update_layout(
            barmode='group', height=380, template='plotly_white',
            yaxis_title='季度周转率（次/季）',
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            annotations=[
                dict(x=mat, y=max(sm.loc[sm[mat_col_sm]==mat, 'turnover_h_quarterly'].values[0],
                                  sm.loc[sm[mat_col_sm]==mat, 'turnover_opt_quarterly'].values[0]) + 0.5,
                     text=f"↓{((sm.loc[sm[mat_col_sm]==mat, 'turnover_opt_quarterly'].values[0] - sm.loc[sm[mat_col_sm]==mat, 'turnover_h_quarterly'].values[0]) / sm.loc[sm[mat_col_sm]==mat, 'turnover_h_quarterly'].values[0] * 100):.1f}%",
                     showarrow=False, font=dict(size=11, color='#e74c3c'))
                for mat in materials_sm
            ]
        )
        st.plotly_chart(fig_to, use_container_width=True)
        
        # 平均库存与资金占用对比
        st.markdown('<p class="section-header">平均库存与资金占用变化</p>', unsafe_allow_html=True)
        
        fig_inv = make_subplots(rows=1, cols=2, subplot_titles=['平均库存量（件）', '库存资金占用（万元）'])
        
        fig_inv.add_trace(go.Bar(
            x=materials_sm, y=sm['avg_inv_h_units'], name='H公司', marker_color='#3498db', width=0.35
        ), row=1, col=1)
        fig_inv.add_trace(go.Bar(
            x=materials_sm, y=sm['avg_inv_opt_units'], name='优化', marker_color='#e67e22', width=0.35
        ), row=1, col=1)
        
        fig_inv.add_trace(go.Bar(
            x=materials_sm, y=sm['avg_inv_value_h'] / WAN, name='H公司', marker_color='#3498db', showlegend=False, width=0.35
        ), row=1, col=2)
        fig_inv.add_trace(go.Bar(
            x=materials_sm, y=sm['avg_inv_value_opt'] / WAN, name='优化', marker_color='#e67e22', showlegend=False, width=0.35
        ), row=1, col=2)
        
        fig_inv.update_layout(barmode='group', height=380, template='plotly_white',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig_inv.update_yaxes(title_text='库存量（件）', row=1, col=1)
        fig_inv.update_yaxes(title_text='资金占用（万元）', row=1, col=2)
        st.plotly_chart(fig_inv, use_container_width=True)
        
        # 缺货成本占比对比
        st.markdown('<p class="section-header">缺货成本占比变化</p>', unsafe_allow_html=True)
        
        fig_so_ratio = go.Figure()
        fig_so_ratio.add_trace(go.Bar(
            x=materials_sm, y=sm['stockout_cost_ratio_h'],
            name='H公司原策略', marker_color='#e74c3c', width=0.35
        ))
        fig_so_ratio.add_trace(go.Bar(
            x=materials_sm, y=sm['stockout_cost_ratio_opt'],
            name='优化策略', marker_color='#2ecc71', width=0.35
        ))
        fig_so_ratio.update_layout(
            barmode='group', height=350, template='plotly_white',
            yaxis_title='缺货成本占运营总成本比例（%）',
            yaxis_tickformat='.1f',
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_so_ratio, use_container_width=True)
        
        # 完整数据表
        st.markdown('<p class="section-header">结构性指标明细</p>', unsafe_allow_html=True)
        display_sm = sm.copy()
        for col in display_sm.columns:
            if display_sm[col].dtype in ['float64', 'float32']:
                display_sm[col] = display_sm[col].round(2)
        st.dataframe(display_sm, use_container_width=True, hide_index=True)
    
    # 仿真参数说明
    st.markdown("---")
    st.markdown('<p class="section-header">仿真参数说明</p>', unsafe_allow_html=True)
    
    params_info = pd.DataFrame({
        '参数': ['仿真次数', '仿真周期', '采购成本口径', '初始库存', '需求分布', '提前期分布', '缺货模式'],
        '取值': ['5000次', '91天（1季度）', '需求驱动(demand×单价)', 'H公司实际值', '正态分布', '经验分布（来自H公司）', 'Lost Sales'],
        '说明': [
            '蒙特卡洛重复次数',
            '对齐企业季度考核周期',
            '消除缺货少买的虚假节约，公平比较运营效率',
            '采用H公司实际初始库存',
            'N(μ, σ²)，参数由历史数据估计',
            '离散概率分布，来源H公司实际数据',
            '缺货视为销售损失，不累积'
        ]
    })
    st.dataframe(params_info, use_container_width=True, hide_index=True)

# ============================================================
# 页面6: 策略参数详情
# ============================================================
elif page == "📋 策略参数详情":
    st.markdown('<p class="main-title">📋 策略参数详情</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 各材料详细参数（使用MATERIALS_MASTER）
    materials_detail = [
        {
            'name': '碟形弹簧',
            'cluster': 'Ⅰ-1 战略型（平稳）',
            'strategy': '(R,Q) 连续检查策略',
            'params': {
                '周均需求d̄': '13,622件',
                '需求标准差σ_d': '785件/周',
                '变异系数CV': '0.06（平稳）',
                '再订货点R': '15,452件',
                '订货量Q': '19,051件',
                '安全库存SS': '1,830件',
                '服务水平': '99%（z=2.326）',
                '提前期均值': '6.7天（约1周）',
                '单价': '10.57元',
                '存储成本': '1.40元/件/周',
                '缺货成本': '154元/件',
            },
            'why': '高价值+核心零件→最高服务水平99%，(R,Q)策略通过EOQ原理确定固定批量Q，最小化总订货成本，实时监控避免缺货导致产线停线'
        },
        {
            'name': '止推环',
            'cluster': 'Ⅱ-1 瓶颈型（平稳）',
            'strategy': '(R,S) 连续检查策略',
            'params': {
                '周均需求d̄': '2,055件',
                '需求标准差σ_d': '208件/周',
                '变异系数CV': '0.10（平稳）',
                '再订货点R': '4,596件',
                '最高库存S': '8,706件',
                '安全库存SS': '486件',
                '服务水平': '95%（z=1.645）',
                '提前期均值': '13.8天（约2周）',
                '单价': '20.50元',
                '存储成本': '1.40元/件/周',
                '缺货成本': '127元/件',
            },
            'why': '瓶颈型物料+供应风险高→(R,S)策略补至目标水平S，订货量随当前库存自适应匹配，灵活应对小幅需求波动'
        },
        {
            'name': '定位轴套',
            'cluster': 'Ⅳ-1 一般型（平稳）',
            'strategy': '(T,S) 周期检查策略',
            'params': {
                '周均需求d̄': '884件',
                '需求标准差σ_d': '230件/周',
                '变异系数CV': '0.26（平稳）',
                '检查周期T': '21天（3周）',
                '最高库存S': '5,269件',
                '安全库存SS': '849件',
                '服务水平': '95%（z=1.645）',
                '提前期均值': '13.8天（约2周）',
                '单价': '7.38元',
                '存储成本': '1.05元/件/周',
                '缺货成本': '60元/件',
            },
            'why': '低价值+需求平稳→周期检查策略，牺牲一定库存精度（风险暴露期从L延长至T+L=5周），但大幅降低监控成本，成本效益最优'
        },
        {
            'name': '密封圈',
            'cluster': 'Ⅲ-2 杠杆型（非平稳）',
            'strategy': '静动结合策略',
            'params': {
                '周均需求d̄': '1,400件',
                '需求标准差σ_d': '560件/周',
                '变异系数CV': '0.46（非平稳，CV>0.4）',
                '计划周期': '13周（1个季度）',
                '服务水平': '95%',
                '提前期均值': '6.9天（约1周）',
                '单价': '15.50元',
                '存储成本': '1.75元/件/周',
                '缺货成本': '115元/件',
                '订货周次': '第1,3,6,8,10周',
                '策略机制': '静态基准量+动态调整量',
            },
            'why': 'CV=0.46>0.4，需求波动显著→静动结合策略：静态基准量保障供应连续性下限，动态调整量根据ARIMA-LSTM预测偏差实时调整订货量'
        },
    ]
    
    for mat in materials_detail:
        with st.expander(f"🔵 {mat['name']} — {mat['strategy']}"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**聚类**: {mat['cluster']}")
                st.markdown("**策略参数**:")
                for k, v in mat['params'].items():
                    st.markdown(f"- {k}: `{v}`")
            with col2:
                st.markdown(f"**选型依据**:")
                st.markdown(mat['why'])
    
    # 下载
    st.markdown("---")
    st.markdown('<p class="section-header">导出数据</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    if 'model_comparison' in data:
        csv1 = data['model_comparison'].to_csv(index=False).encode('utf-8-sig')
        col1.download_button("📥 模型对比数据", csv1, "model_comparison.csv", "text/csv")
    
    if 'mc_comparison' in data:
        csv2 = data['mc_comparison'].to_csv(index=False).encode('utf-8-sig')
        col2.download_button("📥 蒙卡仿真数据", csv2, "monte_carlo_comparison.csv", "text/csv")
    
    if 'strategy_params' in data:
        csv3 = data['strategy_params'].to_csv(index=False).encode('utf-8-sig')
        col3.download_button("📥 策略参数数据", csv3, "inventory_strategy_params.csv", "text/csv")

# ============================================================
# Page 7: Data Update
# ============================================================
elif page == "🔄 数据更新":
    render_data_update()

# ============================================================
# 页脚
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.8rem;'>
    面向汽车零部件的库存管理决策系统 | 2026年中国大学生机械工程创新创意大赛<br>
    数驱精益，智融创新
</div>
""", unsafe_allow_html=True)
