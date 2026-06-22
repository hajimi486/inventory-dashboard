"""
数据更新页面 - 上传新数据后自动完成统计分析->FCM分类->策略匹配->参数计算->仿真验证
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats as sp_stats

WAN = 10000


def render():
    st.markdown('<p class="main-title">🔄 数据更新与在线决策</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("上传新的需求数据后，系统自动完成：**统计分析 → FCM分类 → 策略匹配 → 参数计算 → 仿真验证**")

    # ---- 第1步：数据上传 ----
    st.markdown('<p class="section-header">第1步：上传需求数据</p>', unsafe_allow_html=True)
    st.markdown("""
    **CSV格式要求**：包含 `week`(周次)和各物料需求列，例如：

    | week | disc_spring | thrust_ring | sleeve | seal_ring |
    |------|-------------|-------------|--------|-----------|
    | 1    | 13500       | 2100        | 900    | 1800      |
    | 2    | 14200       | 1950        | 850    | 2200      |
    """)

    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"], key="data_upload")

    st.markdown("**或手动输入物料参数：**")

    with st.expander("📝 手动输入4种物料参数", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**碟形弹簧**")
            d_mean_1 = st.number_input("周均需求", value=13622, key="d_mean_1")
            d_std_1 = st.number_input("需求标准差(件/周)", value=785, key="d_std_1")
            d_price_1 = st.number_input("单价(元)", value=10.57, key="d_price_1")
            d_order_cost_1 = st.number_input("订货成本(元/次)", value=18650, key="d_order_cost_1")
            d_holding_1 = st.number_input("存储成本(元/件/周)", value=1.40, key="d_holding_1")
            d_stockout_1 = st.number_input("缺货成本(元/件)", value=154, key="d_stockout_1")
            d_leadtime_1 = st.number_input("提前期均值(天)", value=7, key="d_leadtime_1")

        with col_b:
            st.markdown("**止推环**")
            d_mean_2 = st.number_input("周均需求", value=2055, key="d_mean_2")
            d_std_2 = st.number_input("需求标准差(件/周)", value=208, key="d_std_2")
            d_price_2 = st.number_input("单价(元)", value=20.50, key="d_price_2")
            d_order_cost_2 = st.number_input("订货成本(元/次)", value=6500, key="d_order_cost_2")
            d_holding_2 = st.number_input("存储成本(元/件/周)", value=1.40, key="d_holding_2")
            d_stockout_2 = st.number_input("缺货成本(元/件)", value=127, key="d_stockout_2")
            d_leadtime_2 = st.number_input("提前期均值(天)", value=14, key="d_leadtime_2")

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("**定位轴套**")
            d_mean_3 = st.number_input("周均需求", value=884, key="d_mean_3")
            d_std_3 = st.number_input("需求标准差(件/周)", value=230, key="d_std_3")
            d_price_3 = st.number_input("单价(元)", value=7.38, key="d_price_3")
            d_order_cost_3 = st.number_input("订货成本(元/次)", value=4100, key="d_order_cost_3")
            d_holding_3 = st.number_input("存储成本(元/件/周)", value=1.05, key="d_holding_3")
            d_stockout_3 = st.number_input("缺货成本(元/件)", value=60, key="d_stockout_3")
            d_leadtime_3 = st.number_input("提前期均值(天)", value=14, key="d_leadtime_3")

        with col_d:
            st.markdown("**密封圈**")
            d_mean_4 = st.number_input("周均需求", value=1946, key="d_mean_4")
            d_std_4 = st.number_input("需求标准差(件/周)", value=895, key="d_std_4")
            d_price_4 = st.number_input("单价(元)", value=15.50, key="d_price_4")
            d_order_cost_4 = st.number_input("订货成本(元/次)", value=12000, key="d_order_cost_4")
            d_holding_4 = st.number_input("存储成本(元/件/周)", value=1.75, key="d_holding_4")
            d_stockout_4 = st.number_input("缺货成本(元/件)", value=115, key="d_stockout_4")
            d_leadtime_4 = st.number_input("提前期均值(天)", value=7, key="d_leadtime_4")

    # ---- 第2步：统计分析 ----
    st.markdown('<p class="section-header">第2步：需求统计分析</p>', unsafe_allow_html=True)

    materials_info = []

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown(f"**已上传数据**：{len(df_upload)}行 x {len(df_upload.columns)}列")
            st.dataframe(df_upload.head(10), use_container_width=True, hide_index=True)

            mat_cols = [c for c in df_upload.columns if c.lower() not in ['week', 'date', 'time']]
            for col in mat_cols:
                vals = df_upload[col].dropna().values
                if len(vals) > 0:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals, ddof=1)
                    cv = std_val / mean_val if mean_val > 0 else 0
                    materials_info.append({
                        'name': col, 'mean': mean_val, 'std': std_val,
                        'cv': cv, 'demand_type': 'steady' if cv <= 0.4 else 'non-steady',
                        'n_weeks': len(vals)
                    })

            if materials_info:
                stats_df = pd.DataFrame(materials_info)
                stats_df['cv'] = stats_df['cv'].round(4)
                stats_df['mean'] = stats_df['mean'].round(1)
                stats_df['std'] = stats_df['std'].round(1)
                display_df = stats_df[['name', 'mean', 'std', 'cv', 'demand_type', 'n_weeks']].copy()
                display_df.columns = ['material', 'weekly_mean', 'weekly_std', 'CV', 'demand_type', 'weeks']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"File read error: {e}")

    # 手动参数
    if not materials_info and uploaded_file is None:
        manual_materials = [
            {'name': 'disc_spring', 'mean': st.session_state.get('d_mean_1', 13622),
             'std': st.session_state.get('d_std_1', 785), 'price': st.session_state.get('d_price_1', 10.57),
             'order_cost': st.session_state.get('d_order_cost_1', 18650), 'holding': st.session_state.get('d_holding_1', 1.40),
             'stockout': st.session_state.get('d_stockout_1', 154), 'leadtime_days': st.session_state.get('d_leadtime_1', 7)},
            {'name': 'thrust_ring', 'mean': st.session_state.get('d_mean_2', 2055),
             'std': st.session_state.get('d_std_2', 208), 'price': st.session_state.get('d_price_2', 20.50),
             'order_cost': st.session_state.get('d_order_cost_2', 6500), 'holding': st.session_state.get('d_holding_2', 1.40),
             'stockout': st.session_state.get('d_stockout_2', 127), 'leadtime_days': st.session_state.get('d_leadtime_2', 14)},
            {'name': 'sleeve', 'mean': st.session_state.get('d_mean_3', 884),
             'std': st.session_state.get('d_std_3', 230), 'price': st.session_state.get('d_price_3', 7.38),
             'order_cost': st.session_state.get('d_order_cost_3', 4100), 'holding': st.session_state.get('d_holding_3', 1.05),
             'stockout': st.session_state.get('d_stockout_3', 60), 'leadtime_days': st.session_state.get('d_leadtime_3', 14)},
            {'name': 'seal_ring', 'mean': st.session_state.get('d_mean_4', 1946),
             'std': st.session_state.get('d_std_4', 895), 'price': st.session_state.get('d_price_4', 15.50),
             'order_cost': st.session_state.get('d_order_cost_4', 12000), 'holding': st.session_state.get('d_holding_4', 1.75),
             'stockout': st.session_state.get('d_stockout_4', 115), 'leadtime_days': st.session_state.get('d_leadtime_4', 7)},
        ]
        for m in manual_materials:
            cv = m['std'] / m['mean'] if m['mean'] > 0 else 0
            materials_info.append({
                'name': m['name'], 'mean': m['mean'], 'std': m['std'],
                'cv': cv, 'demand_type': 'steady' if cv <= 0.4 else 'non-steady',
                'holding': m['holding'], 'stockout': m['stockout'],
                'order_cost': m['order_cost'], 'price': m['price'],
                'leadtime_days': m['leadtime_days'],
            })
        stats_display = pd.DataFrame([{
            'material': m['name'], 'weekly_mean': f"{m['mean']:.0f}",
            'weekly_std': f"{m['std']:.0f}", 'CV': f"{m['cv']:.4f}",
            'demand_type': m['demand_type']
        } for m in materials_info])
        st.dataframe(stats_display, use_container_width=True, hide_index=True)

    # ---- 第3步：FCM分类与策略匹配 ----
    st.markdown('<p class="section-header">Step 3: FCM Classification & Strategy Matching</p>', unsafe_allow_html=True)

    if materials_info:
        for m in materials_info:
            cv = m['cv']
            price = m.get('price', 10)
            if cv <= 0.1 and price >= 10:
                category, strategy, service_level = "I-Strategic(steady)", "(R,Q) Continuous", 0.99
            elif cv <= 0.2 and price >= 15:
                category, strategy, service_level = "II-Bottleneck(steady)", "(R,S) Continuous", 0.95
            elif cv <= 0.3:
                category, strategy, service_level = "IV-General(steady)", "(T,S) Periodic", 0.95
            elif cv > 0.4:
                category, strategy, service_level = "III-Leverage(non-steady)", "Static-Dynamic", 0.95
            else:
                category, strategy, service_level = "IV-General(mid)", "(T,S) Periodic", 0.95
            m['category'] = category
            m['strategy'] = strategy
            m['service_level'] = service_level

        class_df = pd.DataFrame([{
            'material': m['name'], 'CV': f"{m['cv']:.4f}", 'class': m['category'],
            'strategy': m['strategy'], 'SL': f"{m['service_level']*100:.0f}%"
        } for m in materials_info])
        st.dataframe(class_df, use_container_width=True, hide_index=True)

    # ---- 第4步：策略参数计算 ----
    st.markdown('<p class="section-header">Step 4: Auto Parameter Calculation</p>', unsafe_allow_html=True)

    if materials_info and st.button("🔧 Calculate Parameters", key="calc_params"):
        params_results = []
        for m in materials_info:
            d_bar = m['mean']
            sigma_d = m['std']
            strategy = m['strategy']
            sl = m['service_level']
            L_weeks = m.get('leadtime_days', 7) / 7.0
            z = sp_stats.norm.ppf(sl)

            if strategy == "(R,Q) Continuous":
                SS = z * sigma_d * np.sqrt(L_weeks)
                R = d_bar * L_weeks + SS
                holding = m.get('holding', 1.40)
                order_cost = m.get('order_cost', 18650)
                Q = np.sqrt(2 * d_bar * 52 * order_cost / (holding * 52))
                params_results.append({
                    'material': m['name'], 'strategy': strategy,
                    'SS': f"{SS:.0f}", 'R': f"{R:.0f}", 'Q': f"{Q:.0f}",
                    'SL': f"{sl*100:.0f}%", 'L(weeks)': f"{L_weeks:.1f}"
                })
            elif strategy == "(R,S) Continuous":
                SS = z * sigma_d * np.sqrt(L_weeks)
                R = d_bar * L_weeks + SS
                S = d_bar * (L_weeks + 2) + SS
                params_results.append({
                    'material': m['name'], 'strategy': strategy,
                    'SS': f"{SS:.0f}", 'R': f"{R:.0f}", 'S': f"{S:.0f}",
                    'SL': f"{sl*100:.0f}%", 'L(weeks)': f"{L_weeks:.1f}"
                })
            elif strategy == "(T,S) Periodic":
                T_weeks = 3
                sigma_TL = sigma_d * np.sqrt(T_weeks + L_weeks)
                SS = z * sigma_TL
                S = d_bar * (T_weeks + L_weeks) + SS
                params_results.append({
                    'material': m['name'], 'strategy': strategy,
                    'SS': f"{SS:.0f}", 'T': f"{T_weeks*7:.0f}days", 'S': f"{S:.0f}",
                    'SL': f"{sl*100:.0f}%", 'L(weeks)': f"{L_weeks:.1f}"
                })
            elif strategy == "Static-Dynamic":
                params_results.append({
                    'material': m['name'], 'strategy': strategy,
                    'period': '13 weeks', 'SL': f"{sl*100:.0f}%",
                    'note': 'Static base + dynamic adjustment', 'L(weeks)': f"{L_weeks:.1f}"
                })

        if params_results:
            st.dataframe(pd.DataFrame(params_results), use_container_width=True, hide_index=True)
            st.success("✅ Parameters calculated!")

    # ---- 第5步：蒙特卡洛仿真 ----
    st.markdown('<p class="section-header">Step 5: Monte Carlo Simulation</p>', unsafe_allow_html=True)

    if materials_info:
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            n_sims = st.selectbox("Iterations", [1000, 5000, 10000], index=1, key="n_sims")
        with col_sim2:
            sim_days = st.selectbox("Sim days", [91, 182, 365], index=0, key="sim_days")

        if st.button("🎲 Run Simulation", key="run_mc"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            sim_results = []

            for m_idx, m in enumerate(materials_info):
                d_bar = m['mean']
                sigma_d = m['std']
                strategy = m['strategy']
                sl = m['service_level']
                L_weeks = m.get('leadtime_days', 7) / 7.0
                z = sp_stats.norm.ppf(sl)
                holding = m.get('holding', 1.40)
                stockout = m.get('stockout', 154)
                order_cost = m.get('order_cost', 18650)

                if strategy == "(R,Q) Continuous":
                    SS = z * sigma_d * np.sqrt(L_weeks)
                    R_param = d_bar * L_weeks + SS
                    Q_param = np.sqrt(2 * d_bar * 52 * order_cost / (holding * 52))
                elif strategy == "(R,S) Continuous":
                    SS = z * sigma_d * np.sqrt(L_weeks)
                    R_param = d_bar * L_weeks + SS
                    S_param = d_bar * (L_weeks + 2) + SS
                elif strategy == "(T,S) Periodic":
                    T_w = 3
                    SS = z * sigma_d * np.sqrt(T_w + L_weeks)
                    S_param = d_bar * (T_w + L_weeks) + SS

                # Optimized strategy sim
                total_cost_opt = 0
                for _ in range(n_sims):
                    inv = d_bar * 2
                    tot_h = tot_so = tot_ord = 0
                    for day in range(sim_days):
                        demand = max(0, np.random.normal(d_bar / 7, sigma_d / 7))
                        if inv >= demand:
                            inv -= demand
                        else:
                            tot_so += (demand - inv) * stockout
                            inv = 0
                        tot_h += inv * holding / 7
                        if strategy == "(R,Q) Continuous":
                            if inv <= R_param / 7:
                                tot_ord += order_cost
                                inv += Q_param / 7
                        elif strategy == "(R,S) Continuous":
                            if inv <= R_param / 7:
                                tot_ord += order_cost
                                inv = S_param / 7
                        elif strategy == "(T,S) Periodic":
                            if day % 21 == 0:
                                tot_ord += order_cost
                                inv = S_param / 7
                    total_cost_opt += (tot_h + tot_so + tot_ord) * 365 / sim_days

                # Original strategy sim
                total_cost_h = 0
                R_h = d_bar * 1.0
                Q_h = d_bar * 1.3
                for _ in range(n_sims):
                    inv = d_bar * 2
                    tot_h = tot_so = tot_ord = 0
                    for day in range(sim_days):
                        demand = max(0, np.random.normal(d_bar / 7, sigma_d / 7))
                        if inv >= demand:
                            inv -= demand
                        else:
                            tot_so += (demand - inv) * stockout
                            inv = 0
                        tot_h += inv * holding / 7
                        if inv <= R_h / 7:
                            tot_ord += order_cost
                            inv += Q_h / 7
                    total_cost_h += (tot_h + tot_so + tot_ord) * 365 / sim_days

                avg_opt = total_cost_opt / n_sims
                avg_h = total_cost_h / n_sims
                saving = avg_h - avg_opt
                saving_rate = saving / avg_h * 100 if avg_h > 0 else 0

                sim_results.append({
                    'material': m['name'], 'strategy': strategy,
                    'orig_cost': f"{avg_h:,.0f}", 'opt_cost': f"{avg_opt:,.0f}",
                    'saving': f"{saving:,.0f}", 'rate': f"{saving_rate:.1f}%"
                })
                progress = (m_idx + 1) / len(materials_info)
                progress_bar.progress(progress)
                status_text.text(f"Progress: {m_idx+1}/{len(materials_info)} - {m['name']} done")

            progress_bar.empty()
            status_text.empty()

            if sim_results:
                sim_df = pd.DataFrame(sim_results)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

                fig = go.Figure()
                names = [r['material'] for r in sim_results]
                h_costs = [float(r['orig_cost'].replace(',', '')) for r in sim_results]
                opt_costs = [float(r['opt_cost'].replace(',', '')) for r in sim_results]
                fig.add_trace(go.Bar(name='Original', x=names, y=[c / WAN for c in h_costs], marker_color='#e74c3c'))
                fig.add_trace(go.Bar(name='Optimized', x=names, y=[c / WAN for c in opt_costs], marker_color='#27ae60'))
                fig.update_layout(barmode='group', title='Annual Cost Comparison',
                                  yaxis_title='Cost (wan yuan)', template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Simulation complete! Adjust parameters and re-run as needed.")
