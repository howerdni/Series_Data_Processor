import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="透视表生成工具 (Pivot Table Generator)", layout="wide")

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
                st.error("无法解析时间列。请指定正确的时间格式 (Failed to parse time column. Please specify correct time format.)")
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

st.title("透视表生成工具 (Pivot Table Generator)")

st.header("输入参数 (Input Parameters)")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("上传CSV文件 (Upload CSV File)", type=["csv"])
    time_col = st.text_input("时间列名称 (Time Column Name, e.g., Time)", value="Time")
    data_col = st.text_input("数据列名称 (Data Column Name, e.g., value)", value="value")

with col2:
    time_format = st.text_input("时间格式 (Time Format, e.g., %Y/%m/%d %H:%M)", value="%Y/%m/%d %H:%M")
    granularity = st.selectbox("时间粒度 (Time Granularity)", options=["15min", "1h"], index=0)
    agg_method = st.selectbox("聚合方法 (Aggregation Method)", options=["average", "max", "min"], index=0)

if st.button("生成透视表 (Generate Pivot Table)"):
    if not uploaded_file:
        st.error("请先上传一个CSV文件！(Please upload a CSV file first!)")
    elif not time_col or not data_col:
        st.error("必须指定时间和数据列名称！(Time and Data column names must be specified!)")
    else:
        with st.spinner("正在生成透视表... (Generating pivot table...)"):
            pivot_df = process_pivot(uploaded_file, time_col, data_col, time_format, granularity, agg_method)
            if pivot_df is not None:
                st.session_state['pivot_df'] = pivot_df
                st.success("透视表生成成功！(Pivot table generated successfully!)")

if 'pivot_df' in st.session_state and st.session_state['pivot_df'] is not None:
    st.header("透视表结果 (Pivot Table Result)")
    st.dataframe(st.session_state['pivot_df'], use_container_width=True)

    csv = st.session_state['pivot_df'].to_csv(encoding='utf-8-sig')
    st.download_button(
        label="保存透视表 (Save Pivot Table)",
        data=csv,
        file_name=f"pivot_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.markdown("""
---
**关于 (About)**  
欢迎使用透视表生成工具！此工具用于将时间序列数据转换为透视表，支持按时间粒度（15分钟或1小时）进行聚合（平均、最大、最小）。  
(Welcome to the Pivot Table Generator! This tool converts time series data into pivot tables, supporting aggregation (average, max, min) by time granularity (15 minutes or 1 hour).)
""")
