"""
Streamlit 可视化决策看板 —— 面向汽车零部件的库存管理优化系统
运行方式: streamlit run E:\库存优化竞赛\dashboard\app.py
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
    page_title="库存管理优化决策看板",
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
    
    return data

data = load_data()

# ============================================================
# 侧边栏
# ============================================================
st.sidebar.markdown("## 📊 库存管理优化系统")
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
    st.markdown('<p class="main-title">📊 库存管理优化决策看板</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 顶部关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    if 'mc_comparison' in data:
        mc = data['mc_comparison']
        # 优先使用需求采购口径
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
                <div class="metric-label">H公司原年总成本</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{format_currency(total_opt)}</div>
                <div class="metric-label">优化后年总成本</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);">
                <div class="metric-value">{format_currency(total_saving)}</div>
                <div class="metric-label">年节约总额</div>
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
                yaxis_title='年总成本（万元）',
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
    if 'model_comparison' in data:
        st.markdown('<p class="section-header">预测模型性能对比</p>', unsafe_allow_html=True)
        model_df = data['model_comparison']
        st.dataframe(model_df, use_container_width=True, hide_index=True)

# ============================================================
# 页面2: 需求预测分析
# ============================================================
elif page == "📈 需求预测分析":
    st.markdown('<p class="main-title">📈 需求预测分析</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # ARIMA分析
    if 'seal_ring' in data:
        st.markdown('<p class="section-header">密封圈130周需求序列</p>', unsafe_allow_html=True)
        
        seal = data['seal_ring']
        # 尝试找数值列
        num_cols = seal.select_dtypes(include=[np.number]).columns.tolist()
        week_col = 'week' if 'week' in seal.columns else None
        if num_cols:
            fig = px.line(seal, x=week_col, y=num_cols[-1] if 'demand' not in seal.columns else 'demand',
                         title='密封圈历史需求序列（130周）')
            fig.update_layout(template='plotly_white', height=400, xaxis_title='周', yaxis_title='需求量')
            st.plotly_chart(fig, use_container_width=True)
    
    # 多模型预测对比
    if 'model_comparison' in data:
        st.markdown('<p class="section-header">多模型性能指标对比</p>', unsafe_allow_html=True)
        
        mdf = data['model_comparison']
        
        # 找RMSE列
        rmse_col = None
        for c in mdf.columns:
            if 'rmse' in c.lower():
                rmse_col = c
                break
        
        if rmse_col:
            model_col = mdf.columns[0]
            fig = make_subplots(rows=1, cols=3,
                               subplot_titles=['RMSE', 'MAPE', 'R²'])
            
            # RMSE
            fig.add_trace(go.Bar(x=mdf[model_col], y=mdf[rmse_col], 
                                name='RMSE', marker_color='#3498db'), row=1, col=1)
            
            # MAPE
            mape_col = None
            for c in mdf.columns:
                if 'mape' in c.lower():
                    mape_col = c
                    break
            if mape_col:
                fig.add_trace(go.Bar(x=mdf[model_col], y=mdf[mape_col], 
                                    name='MAPE', marker_color='#e67e22'), row=1, col=2)
            
            # R²
            r2_col = None
            for c in mdf.columns:
                if 'r2' in c.lower() or 'r²' in c.lower():
                    r2_col = c
                    break
            if r2_col:
                fig.add_trace(go.Bar(x=mdf[model_col], y=mdf[r2_col], 
                                    name='R²', marker_color='#2ecc71'), row=1, col=3)
            
            fig.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(mdf, use_container_width=True, hide_index=True)
    
    # 13周预测值
    if 'model_forecast' in data:
        st.markdown('<p class="section-header">13周预测值对比</p>', unsafe_allow_html=True)
        fc_df = data['model_forecast']
        st.dataframe(fc_df, use_container_width=True, hide_index=True)
    
    # ARIMA预测详情
    if 'arima_forecast' in data:
        st.markdown('<p class="section-header">ARIMA密封圈预测详情</p>', unsafe_allow_html=True)
        af = data['arima_forecast']
        
        # 找实际值和预测值列
        actual_col = None
        forecast_col = None
        for c in af.columns:
            if 'actual' in c.lower() or '真实' in c:
                actual_col = c
            if 'forecast' in c.lower() or '预测' in c:
                forecast_col = c
        
        if actual_col and forecast_col:
            x_col = 'week' if 'week' in af.columns else af.index
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=af[x_col] if isinstance(x_col, str) else af.index, 
                                    y=af[actual_col], mode='lines+markers', 
                                    name='实际值', line=dict(color='#2c3e50')))
            fig.add_trace(go.Scatter(x=af[x_col] if isinstance(x_col, str) else af.index, 
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
    st.markdown('<p class="section-header">聚类结果</p>', unsafe_allow_html=True)
    
    cluster_data = {
        '材料': ['碟形弹簧', '止推环', '定位轴套', '密封圈'],
        '聚类': ['类别A（高价值平稳）', '类别A（高价值平稳）', '类别C（非平稳）', '类别B（中等波动）'],
        '隶属度': [0.984, 0.377, 0.989, 0.983],
        '推荐策略': ['(R,Q) 连续检查', '(R,Q) 连续检查', '静动结合策略', '(R,S) 连续检查']
    }
    cluster_df = pd.DataFrame(cluster_data)
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)
    
    # 特征矩阵
    st.markdown('<p class="section-header">需求特征矩阵</p>', unsafe_allow_html=True)
    feature_data = {
        '材料': ['碟形弹簧', '止推环', '定位轴套', '密封圈'],
        '均值需求': [13659.5, 2052.2, 901.2, 1946.2],
        '标准差': [778.2, 232.8, 260.7, 192.6],
        '变异系数CV': [0.057, 0.113, 0.289, 0.099],
        '趋势强度': [0.306, 0.313, 0.183, 0.101],
        '斜率': [63.7, 19.5, -12.7, 5.2]
    }
    feature_df = pd.DataFrame(feature_data)
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    # 雷达图
    st.markdown('<p class="section-header">FCM聚类特征雷达图</p>', unsafe_allow_html=True)
    
    radar_path = os.path.join(RESULT_FIG_DIR, "fcm_radar.png")
    if os.path.exists(radar_path):
        st.image(radar_path, use_container_width=True)
    else:
        # 用plotly画
        categories = ['均值需求', '标准差', '变异系数CV', '趋势强度', '斜率']
        materials = ['碟形弹簧', '止推环', '定位轴套', '密封圈']
        raw_vals = [
            [13659.5, 778.2, 0.057, 0.306, 63.7],
            [2052.2, 232.8, 0.113, 0.313, 19.5],
            [901.2, 260.7, 0.289, 0.183, -12.7],
            [1946.2, 192.6, 0.099, 0.101, 5.2],
        ]
        colors_r = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        # 归一化到0-1
        arr = np.array(raw_vals)
        arr_min = arr.min(axis=0)
        arr_max = arr.max(axis=0)
        arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-10)
        
        fig = go.Figure()
        for i, mat in enumerate(materials):
            fig.add_trace(go.Scatterpolar(
                r=arr_norm[i].tolist() + [arr_norm[i][0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=mat,
                opacity=0.6,
                line=dict(color=colors_r[i])
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                         showlegend=True, height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
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
    st.markdown('<p class="section-header">FCM聚类 → 策略映射</p>', unsafe_allow_html=True)
    
    strategy_map = pd.DataFrame({
        '聚类类别': ['类别A（高价值平稳）', '类别B（中等波动）', '类别C（非平稳）'],
        '特征描述': ['需求平稳、价值高、变异系数低', '需求中等波动、价值适中', '需求不稳定、变异系数高'],
        '推荐策略': ['(R,Q) 连续检查策略', '(R,S) 连续检查策略', '静动结合策略'],
        '策略原理': [
            '固定订货量Q，库存降至R即触发订货',
            '库存降至R即订货至最高水平S',
            '平稳期用固定参数，波动期动态调整订货时点和订货量'
        ]
    })
    st.dataframe(strategy_map, use_container_width=True, hide_index=True)
    
    # 策略参数
    st.markdown('<p class="section-header">各材料库存策略参数</p>', unsafe_allow_html=True)
    
    if 'strategy_params' in data:
        st.dataframe(data['strategy_params'], use_container_width=True, hide_index=True)
    else:
        # 硬编码参数表
        params_data = {
            '材料': ['碟形弹簧', '止推环', '定位轴套', '密封圈'],
            '策略类型': ['(R,Q)', '(R,S)', '(T,S)静动结合', '(R,S)'],
            '安全库存ss': [6316, 826, 1038, '动态调整'],
            '订货点R': [24624, 6477, '—', '动态调整'],
            '订货量Q/最高库存S': [19051, 8532, 7182, '动态调整'],
            '检查周期T': ['—', '—', '21天', '—'],
            '服务水平': ['99%', '95%', '95%', '95%']
        }
        st.dataframe(pd.DataFrame(params_data), use_container_width=True, hide_index=True)
    
    # H公司原参数 vs 优化参数对比
    st.markdown('<p class="section-header">原参数 vs 优化参数对比</p>', unsafe_allow_html=True)
    
    comparison = pd.DataFrame({
        '材料': ['碟形弹簧', '止推环', '定位轴套', '密封圈'],
        '原策略': ['(R,Q)', '(R,Q)', '(R,Q)', '(R,Q)'],
        '原订货点R': [14260, 3970, 1775, 2180],
        '优化策略': ['(R,Q)', '(R,S)', '(T,S)', '静动结合'],
        '优化关键参数': ['R=15452,Q=19051', 'R=4596,S=8706', 'T=21天,S=5269', '周次[1,3,6,8,10]'],
        '服务水平': ['99%', '95%', '95%', '95%']
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    
    # 月度库存趋势
    if 'monthly_inv' in data:
        st.markdown('<p class="section-header">H公司2024年月度库存占用</p>', unsafe_allow_html=True)
        mi = data['monthly_inv']
        st.dataframe(mi, use_container_width=True, hide_index=True)

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
        
        # 检查是否有分项数据
        has_breakdown = all(c in mc.columns for c in ['h_holding', 'h_ordering', 'h_stockout'])
        
        if has_breakdown:
            materials = mc[material_col].tolist()
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=['H公司原策略', '优化策略'])
            
            # H公司
            fig.add_trace(go.Bar(x=materials, y=mc['h_holding'] / WAN, name='持有成本', 
                                marker_color='#3498db'), row=1, col=1)
            fig.add_trace(go.Bar(x=materials, y=mc['h_ordering'] / WAN, name='订货成本', 
                                marker_color='#f39c12'), row=1, col=1)
            fig.add_trace(go.Bar(x=materials, y=mc['h_stockout'] / WAN, name='缺货成本', 
                                marker_color='#e74c3c'), row=1, col=1)
            
            # 优化
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
        else:
            # 无分项数据，简单总成本对比
            h_col = 'h_company_cost' if 'h_company_cost' in mc.columns else mc.columns[1]
            opt_col = 'optimized_cost' if 'optimized_cost' in mc.columns else mc.columns[2]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='H公司', x=mc[material_col], y=mc[h_col] / WAN, 
                                marker_color='#e74c3c'))
            fig.add_trace(go.Bar(name='优化后', x=mc[material_col], y=mc[opt_col] / WAN, 
                                marker_color='#27ae60'))
            fig.update_layout(barmode='group', height=450, template='plotly_white',
                            yaxis_title='年总成本（万元）')
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
    
    # 仿真参数说明
    st.markdown("---")
    st.markdown('<p class="section-header">仿真参数说明</p>', unsafe_allow_html=True)
    
    params_info = pd.DataFrame({
        '参数': ['仿真次数', '仿真周期', '采购成本口径', '初始库存', '需求分布', '提前期分布', '缺货模式'],
        '取值': ['5000次', '91天（1季度）', '需求驱动(demand×单价)', '论文实际值', '正态分布', '经验分布（来自论文）', 'Lost Sales'],
        '说明': [
            '蒙特卡洛重复次数',
            '与论文一致的季度仿真周期',
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
    
    # 各材料详细参数
    materials_detail = [
        {
            'name': '碟形弹簧',
            'cluster': '类别A（高价值平稳）',
            'strategy': '(R,Q) 连续检查策略',
            'params': {
                'H公司订货点R': 14260,
                'H公司订货量Q': 17000,
                '优化订货点R': 15452,
                '优化订货量Q': 19051,
                '服务水平': '99%',
                '提前期均值': '6.7天',
                '单价': '10.57元',
                '日均需求': 1946
            },
            'why': '高价值+高需求+低波动→高服务水平+连续检查，避免缺货导致高额停产损失'
        },
        {
            'name': '止推环',
            'cluster': '类别A（高价值平稳）',
            'strategy': '(R,S) 连续检查策略',
            'params': {
                'H公司订货点R': 3970,
                'H公司订货量Q': 4200,
                '优化订货点R': 4596,
                '优化最高库存S': 8706,
                '服务水平': '95%',
                '提前期均值': '13.8天',
                '单价': '20.50元',
                '日均需求': 294
            },
            'why': '中等价值+中等波动→95%服务水平，(R,S)策略订至最高水平S简化补货决策'
        },
        {
            'name': '定位轴套',
            'cluster': '类别C（非平稳）',
            'strategy': '(T,S) 定期检查策略',
            'params': {
                'H公司订货点R': 1775,
                'H公司订货量Q': 2000,
                '优化检查周期T': '21天',
                '优化最高库存S': 5269,
                '服务水平': '95%',
                '提前期均值': '13.8天',
                '单价': '7.38元',
                '日均需求': 126
            },
            'why': '低价值+高波动+需求不稳定→定期检查策略，平稳期用固定周期和目标库存'
        },
        {
            'name': '密封圈',
            'cluster': '类别C（非平稳）',
            'strategy': '静动结合策略',
            'params': {
                'H公司订货点R': 2180,
                'H公司订货量Q': 2800,
                '优化订货周次': '[1,3,6,8,10]',
                '优化订货量': '动态调整',
                '服务水平': '95%',
                '提前期均值': '7.0天',
                '单价': '15.50元',
                '日均需求': 278
            },
            'why': '高波动(CV=0.46>0.4)→静动结合，平稳期用固定参数，波动期动态调整订货时点和订货量'
        }
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
    面向汽车零部件的库存管理优化系统 | 2026年中国大学生机械工程创新创意大赛<br>
    数驱精益，智融创新
</div>
""", unsafe_allow_html=True)
