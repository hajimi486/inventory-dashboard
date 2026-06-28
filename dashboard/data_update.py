"""
数据更新页面 - 上传新数据后自动完成统计分析->FCM分类->策略匹配->参数计算->仿真验证
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from scipy import stats as sp_stats

WAN = 10000

# Result paths for cross-page data sharing
RESULT_TABLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables")


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
            d_mean_4 = st.number_input("周均需求", value=1400, key="d_mean_4")
            d_std_4 = st.number_input("需求标准差(件/周)", value=560, key="d_std_4")
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
    st.markdown('<p class="section-header">第3步：FCM分类与策略匹配</p>', unsafe_allow_html=True)

    if materials_info:
        for m in materials_info:
            cv = m['cv']
            price = m.get('price', 10)
            # FCM分类：基于变异系数CV + 价值特征，与报告表6一致
            # Ⅰ-战略型: 高价值+平稳需求 → (R,Q)  SL=99%
            # Ⅱ-瓶颈型: 中高价值+平稳需求 → (R,S)  SL=95%
            # Ⅲ-杠杆型: 非平稳需求(CV>=0.4) → 静动结合  SL=95%
            # Ⅳ-一般型: 低价值+平稳需求 → (T,S)  SL=95%
            if cv >= 0.4:
                category, strategy, service_level = "Ⅲ-杠杆型（非平稳）", "静动结合", 0.95
            elif cv <= 0.1 and price >= 10:
                category, strategy, service_level = "Ⅰ-战略型（平稳）", "(R,Q)", 0.99
            elif cv <= 0.2:
                category, strategy, service_level = "Ⅱ-瓶颈型（平稳）", "(R,S)", 0.95
            else:
                category, strategy, service_level = "Ⅳ-一般型（平稳）", "(T,S)", 0.95
            m['category'] = category
            m['strategy'] = strategy
            m['service_level'] = service_level

        class_df = pd.DataFrame([{
            'material': m['name'], 'CV': f"{m['cv']:.4f}", 'class': m['category'],
            'strategy': m['strategy'], 'SL': f"{m['service_level']*100:.0f}%"
        } for m in materials_info])
        st.dataframe(class_df, use_container_width=True, hide_index=True)

    # ---- 第4步：策略参数计算 ----
    st.markdown('<p class="section-header">第4步：策略参数自动计算</p>', unsafe_allow_html=True)

    if materials_info and st.button("🔧 计算策略参数", key="calc_params"):
        params_results = []
        for m in materials_info:
            d_bar = m['mean']
            sigma_d = m['std']
            strategy = m['strategy']
            sl = m['service_level']
            L_weeks = m.get('leadtime_days', 7) / 7.0
            z = sp_stats.norm.ppf(sl)

            if strategy == "(R,Q)":
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
            elif strategy == "(R,S)":
                SS = z * sigma_d * np.sqrt(L_weeks)
                R = d_bar * L_weeks + SS
                S = d_bar * (L_weeks + 2) + SS
                params_results.append({
                    'material': m['name'], 'strategy': strategy,
                    'SS': f"{SS:.0f}", 'R': f"{R:.0f}", 'S': f"{S:.0f}",
                    'SL': f"{sl*100:.0f}%", 'L(weeks)': f"{L_weeks:.1f}"
                })
            elif strategy == "(T,S)":
                T_weeks = 3
                sigma_TL = sigma_d * np.sqrt(T_weeks + L_weeks)
                SS = z * sigma_TL
                S = d_bar * (T_weeks + L_weeks) + SS
                params_results.append({
                    'material': m['name'], 'strategy': strategy,
                    'SS': f"{SS:.0f}", 'T': f"{T_weeks*7:.0f}天", 'S': f"{S:.0f}",
                    'SL': f"{sl*100:.0f}%", 'L(weeks)': f"{L_weeks:.1f}"
                })
            elif strategy == "静动结合":
                params_results.append({
                    'material': m['name'], 'strategy': strategy,
                    'period': '13周', 'SL': f"{sl*100:.0f}%",
                    'note': '静态基准量+动态调整量', 'L(weeks)': f"{L_weeks:.1f}"
                })

        if params_results:
            st.dataframe(pd.DataFrame(params_results), use_container_width=True, hide_index=True)
            st.success("✅ 参数计算完成！")

    # ---- 第5步：蒙特卡洛仿真验证 ----
    st.markdown('<p class="section-header">第5步：蒙特卡洛仿真验证</p>', unsafe_allow_html=True)

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        n_sims = st.selectbox("仿真次数", [1000, 5000, 10000], index=1, key="n_sims")
    with col_sim2:
        sim_days = st.selectbox("仿真天数", [91, 182, 365], index=0, key="sim_days")

    col_out1, col_out2 = st.columns(2)
    with col_out1:
        output_period = st.selectbox("成本输出周期", ["季度（91天）", "半年度（182天）", "年度（365天）"], index=0, key="output_period")
    with col_out2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("💡 *当前使用源代码引擎（5000次×91天），参数选择将在后续版本支持自定义*")

    src_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "src", "03_fcm_inventory_montecarlo.py")

    if st.button("🎲 运行仿真", key="run_mc"):
        with st.spinner(f"正在运行蒙特卡洛仿真（{n_sims}次 × {sim_days}天），请稍候..."):
            import subprocess
            import sys

            # Set matplotlib to non-interactive backend to avoid GUI issues
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'

            try:
                result = subprocess.run(
                    [sys.executable, src_script],
                    capture_output=True, text=True, timeout=300,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    env=env
                )

                if result.returncode != 0:
                    # Write error log for debugging
                    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation_error.log")
                    with open(log_path, 'w', encoding='utf-8') as lf:
                        lf.write(f"=== Return Code: {result.returncode} ===\n")
                        lf.write(f"=== STDOUT ({len(result.stdout)} chars) ===\n")
                        lf.write(result.stdout or "(empty)")
                        lf.write(f"\n=== STDERR ({len(result.stderr)} chars) ===\n")
                        lf.write(result.stderr or "(empty)")
                    err_msg = result.stderr or result.stdout or "未知错误（详见 simulation_error.log）"
                    st.error(f"仿真运行失败（返回码：{result.returncode}）：\n```\n{str(err_msg)[-1500:]}\n```")
                else:
                    # Show key output
                    stdout_lines = result.stdout.strip().split('\n')
                    # Extract key summary lines
                    summary_lines = [l for l in stdout_lines if any(k in l for k in 
                        ['运营成本季度节约', '运营节约:', '各材料节约率', '完成'])]
                    if summary_lines:
                        st.info('\n'.join(summary_lines[:10]))

                    # Verify CSV files were generated
                    mc_csv = os.path.join(RESULT_TABLE_DIR, 'monte_carlo_comparison.csv')
                    sm_csv = os.path.join(RESULT_TABLE_DIR, 'structural_metrics.csv')

                    if not os.path.exists(mc_csv):
                        st.error("仿真完成但未找到 monte_carlo_comparison.csv")
                    else:
                        # Clear cache so other pages pick up new data
                        st.cache_data.clear()

                        # Read and display results
                        mc_df = pd.read_csv(mc_csv, encoding='utf-8-sig')

                        # Summary metrics
                        total_saving = mc_df['op_saving'].sum()
                        st.metric("运营成本季度节约总额", f"¥{total_saving:,.0f}")

                        # Per-material results
                        st.markdown("**各材料仿真结果：**")

                        col1, col2 = st.columns(2)
                        with col1:
                            # Cost comparison table
                            display_df = mc_df[['material', 'h_operational', 'opt_operational', 
                                               'op_saving', 'op_saving_pct']].copy()
                            display_df.columns = ['物料', 'H公司运营成本', '优化运营成本', 
                                                  '节约额', '节约率(%)']
                            st.dataframe(display_df, use_container_width=True, hide_index=True)

                        with col2:
                            # Bar chart
                            fig = go.Figure()
                            names = mc_df['material'].tolist()
                            fig.add_trace(go.Bar(name='H公司原策略', x=names, 
                                                y=[c/WAN for c in mc_df['h_operational']], 
                                                marker_color='#e74c3c'))
                            fig.add_trace(go.Bar(name='优化策略', x=names, 
                                                y=[c/WAN for c in mc_df['opt_operational']], 
                                                marker_color='#27ae60'))
                            fig.update_layout(barmode='group', title='运营成本对比（季度）',
                                             yaxis_title='成本（万元）', 
                                             template='plotly_white', height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        # Structural metrics
                        if os.path.exists(sm_csv):
                            sm_df = pd.read_csv(sm_csv, encoding='utf-8-sig')
                            st.markdown("**结构性指标：**")
                            st.dataframe(sm_df, use_container_width=True, hide_index=True)

                        st.success(f"✅ 仿真完成！运营成本季度节约 ¥{total_saving:,.0f}，结果已同步到所有页面。")

            except subprocess.TimeoutExpired:
                st.error("仿真超时（>5分钟），请检查源代码是否正常运行")
            except Exception as e:
                st.error(f"运行异常：{str(e)}")
