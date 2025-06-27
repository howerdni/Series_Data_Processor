import streamlit as st
import pandas as pd
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

st.title("透视表与时间序列转换工具 (Pivot and Unpivot Tool)")

# 创建两个标签页
tab1, tab2 = st.tabs(["透视表 (Pivot Table)", "时间序列转换 (Unpivot Time Series)"])

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

st.markdown("""
---
**关于 (About)**  
欢迎使用透视表与时间序列转换工具！此工具支持将时间序列数据转换为透视表，或将透视表格式数据转换回时间序列格式。  
- 透视表：按时间粒度（15分钟或1小时）聚合数据（平均、最大、最小），支持多种时间格式选择。  
- 时间序列转换：将透视表格式数据转换为时间序列，支持CSV和Excel输入。  
(Welcome to the Pivot and Unpivot Tool! This tool supports converting time series data into pivot tables or transforming pivot table formatted data back to time series format.  
- Pivot Table: Aggregate data (average, max, min) by time granularity (15 minutes or 1 hour), with multiple time format options.  
- Unpivot Time Series: Convert pivot table data to time series, supporting CSV and Excel inputs.)
""")
