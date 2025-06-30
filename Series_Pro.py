import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import re
import os

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="数据分析与负荷预测工具", layout="wide")

class DataAnalysisAndLoadForecastingApp:
    def __init__(self):
        self.load_data = None
        self.time_data = None
        self.P_forecast_value = None
        self.pivot_df = None
        self.unpivot_df = None

    def validate_timeseries(self, timestamps, expected_freq):
        validation_message = []
        is_valid = True

        timestamps = timestamps.dropna()
        if expected_freq == "15T":
            expected_rows = 23232
            if len(timestamps) > expected_rows * 1.5 or len(timestamps) < expected_rows * 0.5:
                is_valid = False
                validation_message.append(f"意外的行数: {len(timestamps)}。预期约{expected_rows}行，适用于15T频率")

        supported_formats = [
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ]
        
        try:
            parsed_timestamps = pd.Series([pd.NaT] * len(timestamps))
            for fmt in supported_formats:
                temp = pd.to_datetime(timestamps, format=fmt, errors='coerce')
                parsed_timestamps = parsed_timestamps.fillna(temp)
            
            parsed_timestamps = parsed_timestamps.dt.floor('S')
            
            if parsed_timestamps.isna().any():
                invalid_count = parsed_timestamps.isna().sum()
                is_valid = False
                validation_message.append(f"{invalid_count}个时间戳无法解析为日期时间")
            
            timestamps = parsed_timestamps
        except Exception as e:
            is_valid = False
            validation_message.append(f"解析时间戳失败: {str(e)}")
            return is_valid, "\n".join(validation_message), timestamps

        duplicates = timestamps[timestamps.duplicated()]
        if not duplicates.empty:
            is_valid = False
            dup_count = len(duplicates)
            dup_examples = duplicates.head(3).to_list()
            validation_message.append(f"发现{dup_count}个重复时间戳。示例: {dup_examples}")

        timestamps = timestamps.drop_duplicates()
        timestamps = timestamps.sort_values()
        freq = expected_freq
        try:
            expected_range = pd.date_range(start=timestamps.min(), end=timestamps.max(), freq=freq)
        except ValueError as e:
            is_valid = False
            validation_message.append(f"无效频率 '{freq}': {str(e)}")
            return is_valid, "\n".join(validation_message), timestamps
        
        if len(timestamps) != len(expected_range):
            is_valid = False
            missing = set(expected_range) - set(timestamps)
            missing_count = len(missing)
            if missing_count > 0:
                missing_examples = list(missing)[:3]
                validation_message.append(f"发现{missing_count}个缺失时间戳。示例: {missing_examples}")
            if len(timestamps) > len(expected_range):
                extra_count = len(timestamps) - len(expected_range)
                validation_message.append(f"发现{extra_count}个超出预期范围的额外时间戳")

        if len(timestamps) > 1:
            try:
                timestamp_set = set(timestamps)
                gaps = []
                for i in range(1, len(expected_range)):
                    if expected_range[i] not in timestamp_set:
                        prev_timestamp = expected_range[i-1]
                        j = i
                        while j < len(expected_range) and expected_range[j] not in timestamp_set:
                            j += 1
                        next_timestamp = expected_range[j] if j < len(expected_range) else timestamps.max()
                        gaps.append((prev_timestamp, next_timestamp))
                        i = j
                if gaps:
                    is_valid = False
                    gap_message = [f"时间序列不连续，频率为{freq}"]
                    for prev, next_ts in gaps[:3]:
                        gap_message.append(f"{prev} 和 {next_ts} 之间的间隙")
                    validation_message.extend(gap_message)
            except Exception as e:
                is_valid = False
                validation_message.append(f"检查连续性时出错: {str(e)}")

        if is_valid:
            validation_message.append(f"时间序列有效: 所有时间戳均为日期时间格式，频率为{freq}，无缺失或重复条目")

        return is_valid, "\n".join(validation_message), timestamps

    def read_excel_to_dataframe(self, file, sheet_name, header_cell, index_col, data_col, extra_cols, expected_freq):
        row_number = int(''.join(filter(str.isdigit, header_cell)))
        df = pd.read_excel(file, sheet_name=sheet_name, header=row_number-1)
        
        def to_column_index(col):
            if isinstance(col, str) and col.isalpha():
                return ord(col.upper()) - ord('A')
            return int(col)
        
        index_col_idx = to_column_index(index_col)
        data_col_idx = to_column_index(data_col)
        extra_col_indices = [to_column_index(col) for col in extra_cols] if extra_cols else []
        
        max_col = len(df.columns)
        for col_idx in [index_col_idx, data_col_idx] + extra_col_indices:
            if col_idx >= max_col:
                raise ValueError(f"列索引 {col_idx} 超出可用列数 ({max_col})")
        
        time_col_name = df.columns[index_col_idx]
        data_col_name = df.columns[data_col_idx]
        extra_col_names = [df.columns[idx] for idx in extra_col_indices]
        
        selected_cols = [time_col_name, data_col_name] + extra_col_names
        df = df[selected_cols]
        
        validation_result = self.validate_timeseries(df[time_col_name], expected_freq)
        is_valid, validation_message, parsed_timestamps = validation_result

        df[time_col_name] = parsed_timestamps
        df.set_index(time_col_name, inplace=True)
        df.columns = ['value'] + extra_col_names
        
        return df, validation_result

    def process_pivot(self, df, time_col, data_col, time_format, granularity, agg_method):
        if time_col not in df.columns:
            raise ValueError(f"时间列 '{time_col}' 在CSV中未找到")
        if data_col not in df.columns:
            raise ValueError(f"数据列 '{data_col}' 在CSV中未找到")
        
        try:
            df[time_col] = pd.to_datetime(df[time_col], format=time_format)
        except ValueError:
            common_formats = [
                '%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y %H:%M', '%d-%m-%Y %H:%M', '%Y%m%d %H:%M', '%Y%m%d %H:%M:%S'
            ]
            for fmt in common_formats:
                try:
                    df[time_col] = pd.to_datetime(df[time_col], format=fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("无法解析时间列。请指定正确的时间格式")
        
        df.set_index(time_col, inplace=True)
        
        agg_func = {'average': 'mean', 'max': 'max', 'min': 'min'}[agg_method]
        
        if granularity == "15min":
            resampled = df[data_col].resample('15T').agg(agg_func)
            resampled = resampled.reset_index()
            resampled['date'] = resampled[time_col].dt.date
            resampled['time'] = resampled[time_col].dt.strftime('%H:%M')
        else:
            resampled = df[data_col].resample('1H').agg(agg_func)
            resampled = resampled.reset_index()
            resampled['date'] = resampled[time_col].dt.date
            resampled['time'] = resampled[time_col].dt.strftime('%H:00')
        
        pivot_df = resampled.pivot(index='date', columns='time', values=data_col)
        
        return pivot_df

    def unpivot_to_timeseries(self, file, file_type, sheet_name, header_cell, granularity):
        row_number = int(''.join(filter(str.isdigit, header_cell)))
        col_letter = ''.join(filter(str.isalpha, header_cell)).upper()
        col_number = sum((ord(c) - ord('A') + 1) * (26 ** i) for i, c in enumerate(reversed(col_letter))) - 1

        if file_type == 'csv':
            df = pd.read_csv(file, header=row_number-1, index_col=0)
        else:
            df = pd.read_excel(file, sheet_name=sheet_name, header=row_number-1, index_col=0)

        df.columns = [str(col).strip() for col in df.columns]
        
        try:
            df.index = pd.to_datetime(df.index)
            dates = df.index.date
            unique_dates = set(dates)
            expected_dates = set(pd.date_range(start=min(dates), end=max(dates), freq='D').date)
            if len(unique_dates) != len(expected_dates) or unique_dates != expected_dates:
                raise ValueError("索引不包含完整一年的唯一每日日期")
        except ValueError as e:
            raise ValueError(f"无效日期索引: {str(e)}")

        if granularity == "1h":
            expected_times = [f"{h:02d}:00" for h in range(24)]
            expected_count = 24
        else:
            expected_times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 15)]
            expected_count = 96

        actual_columns = list(df.columns)
        if actual_columns != expected_times:
            raise ValueError(f"列必须为{granularity}时间。找到: {actual_columns[:3]}...")
        if len(actual_columns) != expected_count:
            raise ValueError(f"预期{expected_count}个时间列，找到{len(actual_columns)}")

        if not df.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all()).all():
            raise ValueError("数据包含非数字值")

        df_stacked = df.stack().reset_index()
        df_stacked.columns = ['date', 'time', 'value']

        df_stacked['datetime'] = pd.to_datetime(
            df_stacked['date'].astype(str) + ' ' + df_stacked['time'],
            format='%Y-%m-%d %H:%M',
            errors='coerce'
        )

        if df_stacked['datetime'].isna().any():
            raise ValueError("无法解析某些日期时间值")

        result_df = df_stacked[['datetime', 'value']].sort_values('datetime')
        return result_df

    def load_and_validate_forecast(self, df, start_pos, time_col, data_col):
        col_start = ord(start_pos[0].upper()) - ord('A')
        row_start = int(start_pos[1:]) - 1
        if row_start < 0 or col_start < 0:
            raise ValueError("无效的起始位置")

        df.columns = [chr(ord('A') + i) for i in range(len(df.columns))]
        df = df.iloc[row_start:]

        if time_col not in df.columns or data_col not in df.columns:
            raise ValueError("指定的时间列或数据列不存在")
        
        time_data = pd.to_datetime(df[time_col], errors='coerce')
        if time_data.isna().any():
            raise ValueError("时间列包含无效或格式错误的时间")
        
        load_data = pd.to_numeric(df[data_col], errors='coerce')
        if load_data.isna().any():
            raise ValueError("负荷数据列包含非数字值")

        start_time = time_data.min()
        end_time = time_data.max()
        expected_hours = int((end_time - start_time).total_seconds() / 3600) + 1
        actual_hours = len(time_data)

        time_diffs = time_data.diff().dropna()
        non_hourly = time_diffs[time_diffs != timedelta(hours=1)]
        interval_ok = non_hourly.empty

        interval_details = ""
        if not non_hourly.empty:
            interval_details = "非1小时间隔详情:\n"
            for idx in non_hourly.index:
                interval_details += f"  介于 {time_data.loc[idx-1]} 和 {time_data.loc[idx]}: {time_diffs.loc[idx]}\n"

        duplicates = time_data.duplicated().sum()
        maximum = load_data.max()
        sumofvalue = load_data.sum()
        minimum = load_data.min()

        result = (
            f"首时间: {start_time}\n"
            f"末时间: {end_time}\n"
            f"时间间隔: {'1小时' if interval_ok else '存在非1小时间距'}\n"
            f"{interval_details if interval_details else ''}"
            f"预期数据点: {expected_hours}\n"
            f"实际数据点: {actual_hours}\n"
            f"重复时间点: {duplicates}\n"
            f"最大值: {maximum}\n"
            f"求和: {sumofvalue}\n"
            f"最小值: {minimum}\n"
        )
        
        return load_data, time_data, result, interval_ok, actual_hours, expected_hours, duplicates

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

        ecos_output = solver_output.getvalue()
        solver_output.truncate(0)
        solver_output.seek(0)

        if P_forecast_value is None:
            ecos_output += "\nECOS失败，尝试SCS...\n"
            with redirect_stdout(solver_output):
                P_min_new_scaled = P_min * (1 + delta / 2) / scale_factor
                constraints[-2] = P_forecast >= P_min_new_scaled
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.SCS, verbose=True)
                P_forecast_value = P_forecast.value
                solver_status = problem.status
            scs_output = solver_output.getvalue()
            solver_output = ecos_output + scs_output
        else:
            solver_output = ecos_output

        if P_forecast_value is None:
            raise ValueError("优化失败：无法找到可行解，请检查参数或约束")

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

    def main(self):
        st.title("数据分析与负荷预测工具 (Data Analysis and Load Forecasting Tool)")

        tabs = st.tabs(["数据提取", "透视分析", "时间序列转换", "负荷预测", "关于"])

        with tabs[0]:  # 数据提取
            st.header("数据提取 (Data Extraction)")
            uploaded_file = st.file_uploader("选择Excel文件", type=['xlsx', 'xls'])
            sheet_name = st.text_input("工作表名称", value="")
            header_cell = st.text_input("表头单元格 (e.g., A5)", value="")
            index_col = st.text_input("索引列 (e.g., B or 2)", value="")
            data_col = st.text_input("数据列 (e.g., C or 3)", value="")
            extra_cols = st.text_input("额外列 (e.g., B,C,E or 2,3,5)", value="")
            expected_freq = st.selectbox("预期频率", ["H", "15T", "5T", "30T"], index=0)

            if st.button("处理文件"):
                if not uploaded_file:
                    st.error("请先选择一个文件！")
                elif not all([sheet_name, header_cell, index_col, data_col]):
                    st.error("工作表、表头、索引和数据字段必须填写！")
                elif not re.match(r'^[A-Z]+\d+$', header_cell):
                    st.error("表头单元格必须为类似'A5'的格式！")
                else:
                    extra_cols_list = [col.strip() for col in extra_cols.split(',')] if extra_cols else []
                    try:
                        df, validation_result = self.read_excel_to_dataframe(
                            uploaded_file, sheet_name, header_cell, index_col, data_col, extra_cols_list, expected_freq
                        )
                        is_valid, validation_message, _ = validation_result
                        st.text_area("时间序列验证结果", validation_message, height=200)
                        if is_valid:
                            st.success("时间序列有效！")
                        else:
                            st.warning("检测到时间序列问题，请检查验证结果。")
                        
                        st.dataframe(df)
                        csv = df.to_csv(index=True, encoding='utf-8-sig')
                        st.download_button(
                            label="下载处理结果 (CSV)",
                            data=csv,
                            file_name="output.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"处理文件失败: {str(e)}")

        with tabs[1]:  # 透视分析
            st.header("透视分析 (Pivot Analysis)")
            uploaded_file = st.file_uploader("选择CSV文件", type=['csv'], key="pivot")
            time_col = st.text_input("时间列名称 (e.g., Time)", value="Time")
            data_col = st.text_input("数据列名称 (e.g., value)", value="value")
            time_format = st.text_input("时间格式 (e.g., %Y/%m/%d %H:%M)", value="%Y/%m/%d %H:%M")
            granularity = st.selectbox("时间粒度", ["15min", "1h"], index=0)
            agg_method = st.selectbox("聚合方法", ["average", "max", "min"], index=0)

            if st.button("生成透视表"):
                if not uploaded_file:
                    st.error("请先选择一个CSV文件！")
                elif not all([time_col, data_col]):
                    st.error("必须指定时间和数据列名称！")
                else:
                    try:
                        df = pd.read_csv(uploaded_file)
                        self.pivot_df = self.process_pivot(df, time_col, data_col, time_format, granularity, agg_method)
                        st.dataframe(self.pivot_df)
                        st.success("透视表生成成功！")
                        csv = self.pivot_df.to_csv(index=True, encoding='utf-8-sig')
                        st.download_button(
                            label="下载透视表 (CSV)",
                            data=csv,
                            file_name="pivot_table.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"处理透视表失败: {str(e)}")

        with tabs[2]:  # 时间序列转换
            st.header("时间序列转换 (Unpivot Time Series)")
            uploaded_file = st.file_uploader("选择CSV或Excel文件", type=['csv', 'xlsx', 'xls'], key="unpivot")
            file_type = 'csv' if uploaded_file and uploaded_file.name.endswith('.csv') else 'excel'
            sheet_name = st.text_input("工作表名称 (仅Excel)", value="", disabled=file_type=='csv')
            header_cell = st.text_input("表头单元格 (e.g., A5)", value="")
            granularity = st.selectbox("时间粒度", ["15min", "1h"], index=1, key="unpivot_granularity")

            if st.button("转换为时间序列"):
                if not uploaded_file:
                    st.error("请先选择一个文件！")
                elif not header_cell:
                    st.error("必须指定表头单元格！")
                elif not re.match(r'^[A-Z]+\d+$', header_cell):
                    st.error("表头单元格必须为类似'A5'的格式！")
                elif file_type == 'excel' and not sheet_name:
                    st.error("Excel文件必须指定工作表名称！")
                else:
                    try:
                        self.unpivot_df = self.unpivot_to_timeseries(uploaded_file, file_type, sheet_name, header_cell, granularity)
                        st.dataframe(self.unpivot_df)
                        st.success("时间序列数据生成成功！")
                        csv = self.unpivot_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="下载时间序列 (CSV)",
                            data=csv,
                            file_name="timeseries.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"转换为时间序列失败: {str(e)}")

        with tabs[3]:  # 负荷预测
            st.header("负荷预测 (Load Forecasting)")
            uploaded_file = st.file_uploader("选择CSV文件", type=['csv'], key="forecast")
            start_pos = st.text_input("表格起始位置 (e.g., A1)", value="A1")
            time_col = st.text_input("时间列标签 (e.g., B)", value="B")
            data_col = st.text_input("负荷数据列 (e.g., C)", value="C")

            if st.button("加载并验证数据"):
                if not uploaded_file:
                    st.error("请先选择CSV文件！")
                else:
                    try:
                        df = pd.read_csv(uploaded_file)
                        self.load_data, self.time_data, result, interval_ok, actual_hours, expected_hours, duplicates = self.load_and_validate_forecast(df, start_pos, time_col, data_col)
                        st.text_area("验证结果", result, height=200)
                        if not interval_ok:
                            st.warning("时间序列应具有1小时间距！")
                        if actual_hours != expected_hours:
                            st.warning(f"数据点数量不匹配！预期 {expected_hours}，实际 {actual_hours}")
                        if duplicates > 0:
                            st.warning(f"发现 {duplicates} 个重复时间点！")
                    except Exception as e:
                        st.error(f"加载或验证失败: {str(e)}")

            st.subheader("预测参数")
            P_max_forecast = st.number_input("预测年份最大负荷 (MW)", min_value=0.0, step=0.1)
            E_total_forecast = st.number_input("预测年份总电量 (MWh)", min_value=0.0, step=0.1)
            delta = st.number_input("最小负荷增益率 (%)", min_value=0.0, step=0.1) / 100

            if st.button("运行优化"):
                if self.load_data is None or self.time_data is None:
                    st.error("请先加载并验证数据！")
                else:
                    try:
                        self.P_forecast_value, result, fig = self.run_optimization(self.load_data, self.time_data, P_max_forecast, E_total_forecast, delta)
                        st.text_area("优化结果", result, height=400)
                        st.pyplot(fig)
                        df = pd.DataFrame({
                            "小时": np.arange(1, len(self.P_forecast_value) + 1),
                            "预测负荷_MW": self.P_forecast_value
                        })
                        csv = df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="下载预测结果 (CSV)",
                            data=csv,
                            file_name="forecast_results.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"优化失败: {str(e)}")

        with tabs[4]:  # 关于
            st.header("关于 (About)")
            about_info = (
                "数据分析与负荷预测工具简介\n\n"
                "欢迎使用数据分析与负荷预测工具，这是一款集成了数据处理、分析和电力负荷预测功能的综合性软件，旨在为用户提供高效、便捷的数据管理与预测解决方案。\n"
                "功能概述:\n"
                "1. 数据提取: 从Excel文件中提取时间序列数据，支持自定义表头、索引列和数据列。\n"
                "2. 透视分析: 将时间序列数据转换为透视表，支持按时间粒度（15分钟或1小时）进行聚合。\n"
                "3. 时间序列转换: 将透视表格式的数据转换回时间序列格式。\n"
                "4. 负荷预测: 基于历史负荷数据进行未来负荷预测，使用凸优化算法。\n"
                "开发者信息:\n"
                " 开发者: 倪程捷 (Chengjie Ni)\n"
                " 职位: 高级工程师，中电顾问华东电力设计院有限公司\n"
                " 邮箱: chengjie_ni@163.com\n"
                "版本信息:\n"
                "   - 当前版本: 1.0\n"
                "   - 发布日期: 2025年6月\n"
            )
            st.text_area("关于", about_info, height=400)

if __name__ == "__main__":
    try:
        import pandas
        import openpyxl
        import cvxpy
        import numpy
        import matplotlib
    except ImportError:
        st.error("请先安装所需包: pip install pandas openpyxl cvxpy numpy matplotlib streamlit")
        exit(1)

    app = DataAnalysisAndLoadForecastingApp()
    app.main()
