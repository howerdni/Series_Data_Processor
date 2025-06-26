import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import io
from contextlib import redirect_stdout

class SeriesDataProcessor:
    def __init__(self):
        self.df = None
        self.time_data = None
        self.load_data = None

    def load_data(self, file):
        try:
            if file.name.endswith('.xlsx'):
                self.df = pd.read_excel(file, engine='openpyxl')
            elif file.name.endswith('.csv'):
                self.df = pd.read_csv(file)
            else:
                st.error("不支持的文件格式。请上传 .xlsx 或 .csv 文件。")
                return False
            return True
        except Exception as e:
            st.error(f"读取文件时出错: {str(e)}")
            return False

    def extract_data(self, time_column, load_column):
        if self.df is not None:
            try:
                self.time_data = pd.to_datetime(self.df[time_column])
                self.load_data = self.df[load_column].astype(float).values
                return True
            except Exception as e:
                st.error(f"提取数据时出错: {str(e)}")
                return False
        return False

    def pivot_data(self, row_column, column_column, value_column):
        if self.df is not None:
            try:
                pivot_table = self.df.pivot_table(index=row_column, columns=column_column, values=value_column, aggfunc='sum')
                return pivot_table
            except Exception as e:
                st.error(f"生成数据透视表时出错: {str(e)}")
                return None
        return None

    def unpivot_data(self, pivot_table):
        if pivot_table is not None:
            try:
                unpivoted = pivot_table.reset_index().melt(id_vars=pivot_table.index.name, 
                                                          value_vars=pivot_table.columns,
                                                          var_name=pivot_table.columns.name,
                                                          value_name='Value')
                return unpivoted
            except Exception as e:
                st.error(f"取消数据透视时出错: {str(e)}")
                return None
        return None

    def run_optimization(self, load_data, time_data, P_max_forecast, E_total_forecast, delta):
        P_max_current = np.max(load_data)
        E_total_current = np.sum(load_data)
        P_min = np.min(load_data)
        P_min_new = P_min * (1 + delta)
        n_hours = len(load_data)

        if E_total_forecast < n_hours * P_min_new:
            raise ValueError(f"总电量 {E_total_forecast} MWh 不足以满足最小负荷约束 {n_hours * P_min_new} MWh")
        if E_total_forecast > n_hours * P_max_forecast:
            raise ValueError(f"总电量 {E_total_forecast} MWh 超出最大负荷约束 {n_hours * P_max_forecast} MWh")
        if P_min_new >= P_max_forecast:
            raise ValueError(f"调整后的最小负荷 {P_min_new} MW 不能大于或等于预测最大负荷 {P_max_forecast} MW")

        annual_avg_load = E_total_forecast / n_hours
        scale_factor = 1000
        load_data_scaled = load_data / scale_factor
        P_max_current_scaled = P_max_current / scale_factor
        P_max_forecast_scaled = P_max_forecast / scale_factor
        E_total_forecast_scaled = E_total_forecast / scale_factor
        P_min_new_scaled = P_min_new / scale_factor

        P_scaled = load_data_scaled * (P_max_forecast_scaled / P_max_current_scaled)
        P_forecast = cp.Variable(n_hours)

        objective = cp.Minimize(cp.sum_squares((P_forecast - P_scaled) / P_scaled))
        max_load_idx = np.argmax(load_data)
        constraints = [
            P_forecast[max_load_idx] >= P_max_forecast_scaled * 0.99,
            P_forecast[max_load_idx] <= P_max_forecast_scaled * 1.01,
            P_forecast <= P_max_forecast_scaled,
            cp.sum(P_forecast) == E_total_forecast_scaled,
            P_forecast >= P_min_new_scaled,
            P_forecast >= 0
        ]

        solver_output = io.StringIO()
        solver_status = None
        P_forecast_value = None

        with redirect_stdout(solver_output):
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=True)
            P_forecast_value = P_forecast.value
            solver_status = problem.status

        solver_output = solver_output.getvalue()

        if P_forecast_value is None:
            raise ValueError("优化失败：ECOS无法找到可行解，请检查参数或约束")

        P_forecast_value = P_forecast_value * scale_factor

        result = (
            f"优化算法计算结果:\n"
            f"1. 输入参数:\n"
            f"   - 当前最大负荷: {P_max_current:.2f} MW\n"
            f"   - 当前总电量: {E_total_current:.2f} MWh\n"
            f"   - 当前最小负荷: {P_min:.2f} MW\n"
            f"   - 预测最大负荷: {P_max_forecast:.2f} MW\n"
            f"   - 预测总电量: {E_total_forecast:.2f} MWh\n"
            f"   - 最小负荷增益率: {delta*100:.2f}%\n"
            f"   - 新最小负荷约束: {P_min_new:.2f} MW\n"
            f"   - 年平均负荷: {annual_avg_load:.2f} MW\n"
            f"   - 数据点数: {n_hours}\n"
            f"2. 优化目标:\n"
            f"   - 最小化预测负荷与缩放负荷的相对平方误差和\n"
            f"3. 约束条件:\n"
            f"   - 最大负荷点接近 P_max_forecast (±1%): {P_max_forecast_scaled*0.99*scale_factor:.2f} MW ≤ P[{max_load_idx}] ≤ {P_max_forecast_scaled*1.01*scale_factor:.2f} MW\n"
            f"   - 所有负荷 ≤ P_max_forecast: {P_max_forecast:.2f} MW\n"
            f"   - 总电量 = E_total_forecast: {E_total_forecast:.2f} MWh\n"
            f"   - 所有负荷 ≥ P_min_new: {P_min_new:.2f} MW\n"
            f"   - 所有负荷 ≥ 0 MW\n"
            f"4. CVXPY 求解器日志:\n{solver_output}\n"
            f"5. 求解状态: {solver_status}\n"
            f"6. 输出结果:\n"
            f"   - 预测最大负荷: {np.max(P_forecast_value):.2f} MW\n"
            f"   - 预测总电量: {np.sum(P_forecast_value):.2f} MWh\n"
            f"   - 预测最小负荷: {np.min(P_forecast_value):.2f} MW\n"
            f"优化完成！\n"
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_data, load_data, label='初始负荷数据 (MW)', color='blue', alpha=0.7)
        ax.plot(time_data, P_forecast_value, label='预测负荷数据 (MW)', color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('时间')
        ax.set_ylabel('负荷 (MW)')
        ax.set_title('初始与预测负荷对比')
        ax.legend()
        ax.grid(True)

        return P_forecast_value, result, fig

def main():
    st.title("时间序列数据处理器")
    processor = SeriesDataProcessor()

    tabs = st.tabs(["数据提取", "数据透视", "取消数据透视", "负荷预测"])

    with tabs[0]:
        st.header("数据提取")
        uploaded_file = st.file_uploader("上传Excel或CSV文件", type=['xlsx', 'csv'])
        if uploaded_file is not None:
            if processor.load_data(uploaded_file):
                st.success("文件上传成功！")
                columns = processor.df.columns.tolist()
                time_column = st.selectbox("选择时间列", columns, key="time_column_extract")
                load_column = st.selectbox("选择负荷列", columns, key="load_column_extract")
                if st.button("提取数据"):
                    if processor.extract_data(time_column, load_column):
                        st.success("数据提取成功！")
                        st.write("时间数据预览：")
                        st.write(processor.time_data.head())
                        st.write("负荷数据预览：")
                        st.write(processor.load_data[:5])

    with tabs[1]:
        st.header("数据透视")
        if processor.df is not None:
            columns = processor.df.columns.tolist()
            row_column = st.selectbox("选择行字段", columns, key="row_column_pivot")
            column_column = st.selectbox("选择列字段", columns, key="column_column_pivot")
            value_column = st.selectbox("选择值字段", columns, key="value_column_pivot")
            if st.button("生成数据透视表"):
                pivot_table = processor.pivot_data(row_column, column_column, value_column)
                if pivot_table is not None:
                    st.write("数据透视表：")
                    st.write(pivot_table)
        else:
            st.warning("请先在'数据提取'选项卡中上传并提取数据。")

    with tabs[2]:
        st.header("取消数据透视")
        if processor.df is not None:
            columns = processor.df.columns.tolist()
            row_column = st.selectbox("选择行字段", columns, key="row_column_unpivot")
            column_column = st.selectbox("选择列字段", columns, key="column_column_unpivot")
            value_column = st.selectbox("选择值字段", columns, key="value_column_unpivot")
            if st.button("取消数据透视"):
                pivot_table = processor.pivot_data(row_column, column_column, value_column)
                if pivot_table is not None:
                    unpivoted = processor.unpivot_data(pivot_table)
                    if unpivoted is not None:
                        st.write("取消透视后的数据：")
                        st.write(unpivoted)
        else:
            st.warning("请先在'数据提取'选项卡中上传并提取数据。")

    with tabs[3]:
        st.header("负荷预测")
        if processor.load_data is not None and processor.time_data is not None:
            P_max_forecast = st.number_input("输入预测最大负荷 (MW)", min_value=0.0, value=100.0)
            E_total_forecast = st.number_input("输入预测总电量 (MWh)", min_value=0.0, value=1000.0)
            delta = st.slider("选择最小负荷增益率 (%)", min_value=0.0, max_value=100.0, value=10.0) / 100.0
            if st.button("运行优化"):
                try:
                    P_forecast_value, result, fig = processor.run_optimization(
                        processor.load_data, processor.time_data, P_max_forecast, E_total_forecast, delta
                    )
                    st.write(result)
                    st.pyplot(fig)
                    result_df = pd.DataFrame({
                        '时间': processor.time_data,
                        '预测负荷 (MW)': P_forecast_value
                    })
                    st.write("预测负荷数据：")
                    st.write(result_df)
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="下载预测结果",
                        data=csv,
                        file_name="forecast_results.csv",
                        mime="text/csv"
                    )
                except ValueError as e:
                    st.error(f"优化失败: {str(e)}")
        else:
            st.warning("请先在'数据提取'选项卡中提取时间和负荷数据。")

if __name__ == "__main__":
    main()
