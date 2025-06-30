import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import re

st.set_page_config(page_title="透视表与时间序列转换工具 (Pivot and Unpivot Tool)", layout="wide")

# 定义常见时间格式供用户选择
common_formats = [
    '%Y/%m/%d %H:%M',
    '%Y-%m-%d %H:%M',
    '%Y/%m/%d %H:%M:%S',
    '%Y-%m-%d %H:%M:%S',
    '%d/%m/%Y %H:%M',
    '%d-%m-%Y %H:%M',
    '%Y%m%d %H:%M',
    '%Y%m%d %H:%M:%S'
]

# 透视表处理函数
def process_pivot(file, time_col, data_col, time_format, granularity, agg_method):
    try:
        df = pd.read_csv(file)
        
        if time_col not in df.columns:
            st.error(f"时间列 '{time_col}' 在CSV中未找到 (Time column '{time_col}' not found in CSV)")
            return None
        if data_col not in df.columns:
            st.error(f"数据列 '{data_col}' 在CSV中未找到 (Data column '{data_col}' not found in CSV)")
            return None
        
        try:
            df[time_col] = pd.to_datetime(df[time_col], format=time_format)
        except ValueError:
            st.error(f"无法解析时间列，选定的格式 '{time_format}' 不匹配数据 (Failed to parse time column with selected format '{time_format}').")
            return None
        
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
    except Exception as e:
        st.error(f"处理透视表失败: {str(e)} (Failed to process pivot: {str(e)})")
        return None

# 时间序列转换函数
def process_unpivot(file, file_type, sheet_name, header_cell, granularity):
    try:
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
                raise ValueError("索引不包含完整一年的唯一每日日期 (Index does not contain exactly one year's worth of unique daily dates).")
        except ValueError as e:
            raise ValueError(f"无效日期索引: {str(e)} (Invalid date index: {str(e)})")

        if granularity == "1h":
            expected_times = [f"{h}:00" for h in range(24)]
            alternate_times = [f"0{h}:00" if h < 10 else f"{h}:00" for h in range(24)]
            expected_count = 24
        else:
            expected_times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 15)]
            expected_count = 96

        actual_columns = list(df.columns)
        if actual_columns not in [expected_times, alternate_times]:
            raise ValueError(f"列必须为{granularity}时间（例如 {expected_times[:3]}...）。找到: {actual_columns[:3]}... (Columns must be {granularity} times (e.g., {expected_times[:3]}...). Found: {actual_columns[:3]}...)")
        if len(actual_columns) != expected_count:
            raise ValueError(f"预期{expected_count}个时间列适用于{granularity}频率，找到{len(actual_columns)} (Expected {expected_count} time columns for {granularity} frequency, found {len(actual_columns)})")

        if actual_columns == alternate_times:
            df.columns = expected_times

        if not df.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all()).all():
            raise ValueError("数据包含非数字值 (Data contains non-numeric values).")

        df_stacked = df.stack().reset_index()
        df_stacked.columns = ['date', 'time', 'value']

        df_stacked['datetime'] = pd.to_datetime(
            df_stacked['date'].astype(str) + ' ' + df_stacked['time'],
            format='%Y-%m-%d %H:%M',
            errors='coerce'
        )

        if df_stacked['datetime'].isna().any():
            raise ValueError("无法解析某些日期时间值 (Failed to parse some datetime values).")

        result_df = df_stacked[['datetime', 'value']].sort_values('datetime')

        return result_df
    except Exception as e:
        st.error(f"转换为时间序列失败: {str(e)} (Failed to convert to time series: {str(e)})")
        return None

# 风电出力率分析函数
def get_season(month):
    if month in [3, 4, 5]:
        return '春季'
    elif month in [6, 7, 8]:
        return '夏季'
    elif month in [9, 10, 11]:
        return '秋季'
    elif month in [12, 1, 2]:
        return '冬季'
    return None

def is_in_time_slot(time, time_slot, sample_period):
    hour = time.hour
    if sample_period == 60:
        if time_slot == '19:00-22:00':
            return 19 <= hour <= 22
        elif time_slot == '11:00-14:00':
            return 11 <= hour <= 14
        elif time_slot == '1:00-4:00':
            return 1 <= hour <= 4
    else:
        minute = time.minute
        if time_slot == '19:00-22:00':
            return (19 <= hour < 22) or (hour == 22 and minute == 0)
        elif time_slot == '11:00-14:00':
            return (11 <= hour < 14) or (hour == 14 and minute == 0)
        elif time_slot == '1:00-4:00':
            return (1 <= hour < 4) or (hour == 4 and minute == 0)
    return False

def calculate_power_rate(df, time_col, value_col, unit_col, value_unit, total_capacity, sample_period, percentile):
    time_slots = ['19:00-22:00', '11:00-14:00', '1:00-4:00']
    seasons = ['春季', '夏季', '秋季', '冬季']
    results = []
    
    # 单位转换
    if value_unit == "MWh":
        total_capacity_kW = total_capacity * 1000
        value_to_kW = lambda x: x * 1000
    else:
        total_capacity_kW = total_capacity
        value_to_kW = lambda x: x
    
    # 采样周期（小时）
    period_in_hours = sample_period / 60.0
    
    for season in seasons:
        df_season = df[df[time_col].dt.month.apply(lambda x: get_season(x) == season)]
        
        for slot in time_slots:
            df_slot = df_season[df_season[time_col].apply(lambda x: is_in_time_slot(x, slot, sample_period))]
            
            if not df_slot.empty:
                # 聚合多机组数据
                total_power = df_slot.groupby(time_col)[value_col].sum()
                
                if sample_period != 60:
                    total_power = total_power.resample(f'{sample_period}T').sum()
                
                # 转换为功率（kW）
                total_power = value_to_kW(total_power) / period_in_hours
                power_rate = total_power / total_capacity_kW
                sorted_power_rate = power_rate.sort_values(ascending=False)
                top_percentile = np.percentile(sorted_power_rate, 100 - percentile) if not sorted_power_rate.empty else np.nan
            else:
                top_percentile = np.nan
            
            results.append({
                '季节': season,
                '时段': slot,
                f'{percentile}%概率出力率': top_percentile
            })
    
    return pd.DataFrame(results)

# 主界面
st.title("透视表与时间序列转换工具 (Pivot and Unpivot Tool)")

# 创建三个标签页
tab1, tab2, tab3 = st.tabs(["透视表 (Pivot Table)", "时间序列转换 (Unpivot Time Series)", "风电出力率分析 (Wind Power Analysis)"])

# 透视表标签页
with tab1:
    st.header("透视表生成 (Pivot Table Generation)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pivot_file = st.file_uploader("上传CSV文件 (Upload CSV File)", type=["csv"], key="pivot_file")
        pivot_time_col = st.text_input("时间列名称 (Time Column Name, e.g., Time)", value="Time", key="pivot_time_col")
        pivot_data_col = st.text_input("数据列名称 (Data Column Name, e.g., value)", value="value", key="pivot_data_col")
    
    with col2:
        pivot_time_format = st.selectbox(
            "时间格式 (Time Format)",
            options=common_formats,
            index=0,
            format_func=lambda x: f"{x} (e.g., {datetime.now().strftime(x)})",
            key="pivot_time_format"
        )
        pivot_granularity = st.selectbox("时间粒度 (Time Granularity)", options=["15min", "1h"], index=0, key="pivot_granularity")
        pivot_agg_method = st.selectbox("聚合方法 (Aggregation Method)", options=["average", "max", "min"], index=0, key="pivot_agg_method")
    
    if st.button("生成透视表 (Generate Pivot Table)", key="generate_pivot"):
        if not pivot_file:
            st.error("请先上传一个CSV文件！(Please upload a CSV file first!)")
        elif not pivot_time_col or not pivot_data_col:
            st.error("必须指定时间和数据列名称！(Time and Data column names must be specified!)")
        else:
            with st.spinner("正在生成透视表... (Generating pivot table...)"):
                pivot_df = process_pivot(pivot_file, pivot_time_col, pivot_data_col, pivot_time_format, pivot_granularity, pivot_agg_method)
                if pivot_df is not None:
                    st.session_state['pivot_df'] = pivot_df
                    st.success("透视表生成成功！(Pivot table generated successfully!)")
    
    if 'pivot_df' in st.session_state and st.session_state['pivot_df'] is not None:
        st.subheader("透视表结果 (Pivot Table Result)")
        st.dataframe(st.session_state['pivot_df'], use_container_width=True)
    
        csv = st.session_state['pivot_df'].to_csv(encoding='utf-8-sig')
        st.download_button(
            label="保存透视表 (Save Pivot Table)",
            data=csv,
            file_name=f"pivot_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="save_pivot"
        )

# 时间序列转换标签页
with tab2:
    st.header("时间序列转换 (Unpivot Time Series)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unpivot_file = st.file_uploader("上传CSV或Excel文件 (Upload CSV or Excel File)", type=["csv", "xlsx", "xls"], key="unpivot_file")
        unpivot_sheet_name = st.text_input("工作表名称 (Sheet Name, for Excel)", value="", key="unpivot_sheet_name")
    
    with col2:
        unpivot_header_cell = st.text_input("表头单元格 (Header Cell, e.g., A5)", value="A5", key="unpivot_header_cell")
        unpivot_granularity = st.selectbox("时间粒度 (Time Granularity)", options=["15min", "1h"], index=0, key="unpivot_granularity")
    
    if st.button("转换为时间序列 (Convert to Time Series)", key="generate_unpivot"):
        if not unpivot_file:
            st.error("请先上传一个文件！(Please upload a file first!)")
        elif not unpivot_header_cell:
            st.error("必须指定表头单元格！(Header cell must be specified!)")
        elif not re.match(r'^[A-Z]+\d+$', unpivot_header_cell):
            st.error("表头单元格必须为类似'A5'的格式！(Header cell must be in format like 'A5'!)")
        elif unpivot_file.name.endswith(('.xlsx', '.xls')) and not unpivot_sheet_name:
            st.error("Excel文件必须指定工作表名称！(Sheet name must be specified for Excel files!)")
        else:
            with st.spinner("正在转换为时间序列... (Converting to time series...)"):
                file_type = 'excel' if unpivot_file.name.endswith(('.xlsx', '.xls')) else 'csv'
                unpivot_df = process_unpivot(unpivot_file, file_type, unpivot_sheet_name, unpivot_header_cell, unpivot_granularity)
                if unpivot_df is not None:
                    st.session_state['unpivot_df'] = unpivot_df
                    st.success("时间序列数据生成成功！(Time series data generated successfully!)")
    
    if 'unpivot_df' in st.session_state and st.session_state['unpivot_df'] is not None:
        st.subheader("时间序列结果 (Time Series Result)")
        st.dataframe(st.session_state['unpivot_df'], use_container_width=True)
    
        csv = st.session_state['unpivot_df'].to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="保存时间序列 (Save Time Series)",
            data=csv,
            file_name=f"time_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="save_unpivot"
        )

# 风电出力率分析标签页
with tab3:
    st.header("风电出力率分析 (Wind Power Analysis)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        wind_file = st.file_uploader("上传CSV文件 (Upload CSV File)", type=["csv"], key="wind_file")
        if wind_file:
            try:
                df = pd.read_csv(wind_file, encoding='gbk')
                columns = df.columns.tolist()
                st.write("检测到的列名：", columns)
            except Exception as e:
                st.error(f"读取文件失败: {str(e)}")
                columns = []
        else:
            columns = []
        
        time_col = st.selectbox("时间列", columns, key="wind_time_col", help="选择时间戳列（格式如 2023-03-15 19:00:00）")
        value_col = st.selectbox("值列（发电量）", columns, key="wind_value_col", help="选择发电量列")
        unit_col = st.selectbox("机组编号列", columns, key="wind_unit_col", help="选择机组编号列")
    
    with col2:
        value_unit = st.selectbox("值列单位", ["kWh", "MWh"], key="wind_value_unit", help="发电量数据的单位")
        total_capacity = st.number_input(
            f"总容量（{value_unit}）",
            min_value=0.0,
            value=51.0 if value_unit == "MWh" else 51000.0,
            step=0.1,
            key="wind_total_capacity",
            help="输入所有机组的总容量"
        )
        sample_period_options = {
            "1小时": 60,
            "30分钟": 30,
            "15分钟": 15
        }
        sample_period_label = st.selectbox(
            "采样周期",
            list(sample_period_options.keys()),
            key="wind_sample_period",
            help="选择数据的时间间隔"
        )
        sample_period = sample_period_options[sample_period_label]
        percentile = st.number_input(
            "分位数（%）",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            key="wind_percentile",
            help="例如，5 表示前5%分位出力率"
        )
    
    # 验证采样周期
    if wind_file and time_col and unit_col and columns:
        try:
            df = pd.read_csv(wind_file, encoding='gbk')
            df[time_col] = pd.to_datetime(df[time_col])
            df_sorted = df.sort_values([unit_col, time_col])
            time_diff = df_sorted.groupby(unit_col)[time_col].diff().dropna().dt.total_seconds() / 60
            common_period = time_diff.mode()[0] if not time_diff.empty else None
            if common_period and abs(common_period - sample_period) > 1e-6:
                st.warning(
                    f"检测到数据采样周期约为 {int(common_period)} 分钟，"
                    f"与选择的 {sample_period_label}（{sample_period} 分钟）不匹配，请确认！"
                )
            else:
                st.info(f"采样周期验证通过：约为 {sample_period} 分钟")
        except Exception as e:
            st.error(f"时间列格式错误: {str(e)}")
    
    if st.button("运行分析 (Run Analysis)", key="wind_analyze"):
        if not wind_file:
            st.error("请先上传一个CSV文件！(Please upload a CSV file first!)")
        elif not time_col or not value_col or not unit_col:
            st.error("必须选择时间列、值列和机组编号列！(Time, value, and unit columns must be selected!)")
        elif total_capacity <= 0:
            st.error("总容量必须大于0！(Total capacity must be greater than 0!)")
        else:
            with st.spinner("正在分析数据... (Analyzing data...)"):
                try:
                    df = pd.read_csv(wind_file, encoding='gbk')
                    df[time_col] = pd.to_datetime(df[time_col])
                    results_df = calculate_power_rate(
                        df, time_col, value_col, unit_col, value_unit, total_capacity, sample_period, percentile
                    )
                    st.session_state['wind_df'] = results_df
                    st.success("分析完成！(Analysis completed!)")
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")
    
    if 'wind_df' in st.session_state and st.session_state['wind_df'] is not None:
        st.subheader("分析结果 (Analysis Result)")
        st.dataframe(
            st.session_state['wind_df'].style.format({f'{percentile}%概率出力率': '{:.4f}'}, na_rep='无有效数据'),
            use_container_width=True
        )
        csv = st.session_state['wind_df'].to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="保存结果 (Save Results)",
            data=csv,
            file_name=f"wind_power_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="save_wind"
        )

st.markdown("""
---
**关于 (About)**  
欢迎使用透视表与时间序列转换工具！此工具支持以下功能：  
- **透视表**：将时间序列数据转换为透视表，按时间粒度（15分钟或1小时）聚合数据（平均、最大、最小），支持多种时间格式选择。  
- **时间序列转换**：将透视表格式数据转换为时间序列，支持CSV和Excel输入。  
- **风电出力率分析**：分析风电数据，计算各季节（春季、夏季、秋季、冬季）在特定时段（19:00-22:00、11:00-14:00、1:00-4:00）的指定分位出力率，支持多机组数据和不同采样周期（1小时、30分钟、15分钟）。  
(Welcome to the Pivot and Unpivot Tool! This tool supports the following features:  
- **Pivot Table**: Convert time series data into pivot tables, aggregating data (average, max, min) by time granularity (15 minutes or 1 hour), with multiple time format options.  
- **Unpivot Time Series**: Convert pivot table data to time series, supporting CSV and Excel inputs.  
- **Wind Power Analysis**: Analyze wind power data to calculate percentile power rates for seasons (spring, summer, autumn, winter) in specific time slots (19:00-22:00, 11:00-14:00, 1:00-4:00), supporting multi-unit data and various sampling periods (1 hour, 30 minutes, 15 minutes).)
""")
