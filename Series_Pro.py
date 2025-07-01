import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
import csv
import io

st.set_page_config(page_title="时间序列数据转换与分析工具 (Time Series Data Conversion and Analysis Tool)", layout="wide")

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
            st.error(f"时间列 '{time_col}' 未在 CSV 文件中找到")
            return None
        if data_col not in df.columns:
            st.error(f"数据列 '{data_col}' 未在 CSV 文件中找到")
            return None
        
        try:
            df[time_col] = pd.to_datetime(df[time_col], format=time_format)
        except ValueError:
            st.error(f"无法解析时间列，选定的格式 '{time_format}' 与数据不匹配")
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
        st.error(f"处理透视表失败: {str(e)}")
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
                raise ValueError("索引不包含完整一年的唯一每日日期")
        except ValueError as e:
            raise ValueError(f"无效日期索引: {str(e)}")

        if granularity == "1h":
            expected_times = [f"{h}:00" for h in range(24)]
            alternate_times = [f"0{h}:00" if h < 10 else f"{h}:00" for h in range(24)]
            expected_count = 24
        else:
            expected_times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 15)]
            expected_count = 96

        actual_columns = list(df.columns)
        if actual_columns not in [expected_times, alternate_times]:
            raise ValueError(f"列必须为{granularity}时间（例如 {expected_times[:3]}...）。找到: {actual_columns[:3]}...")
        if len(actual_columns) != expected_count:
            raise ValueError(f"预期{expected_count}个时间列适用于{granularity}频率，找到{len(actual_columns)}")

        if actual_columns == alternate_times:
            df.columns = expected_times

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
    except Exception as e:
        st.error(f"时间序列转换失败: {str(e)}")
        return None

# 风电出力率分析函数
def get_season(month):
    if month == '全年':
        return '全年'
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
    seasons = ['春季', '夏季', '秋季', '冬季', '全年']
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
        if season == '全年':
            df_season = df
        else:
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
                '季节': season.strip(),
                '时段': slot.strip(),
                f'{percentile}%概率出力率': top_percentile
            })
    
    results_df = pd.DataFrame(results)
    # 确保列名无多余空格
    results_df.columns = [col.strip() for col in results_df.columns]
    return results_df

# 主界面
st.title("时间序列数据转换与分析工具 (Time Series Data Conversion and Analysis Tool)")

# 创建四个标签页
tab1, tab2, tab3, tab4 = st.tabs(["透视表 (Pivot Table)", "时间序列转换 (Unpivot Time Series)", "风电出力率分析 (Wind Power Analysis)", "关于 (About)"])

# 透视表标签页
with tab1:
    st.header("透视表生成")
    
    # 初始化 session state 以保存时间格式
    if 'pivot_time_format' not in st.session_state:
        st.session_state['pivot_time_format'] = common_formats[0]  # 默认格式
    
    col1, col2 = st.columns(2)
    
    with col1:
        pivot_file = st.file_uploader("上传 CSV 文件", type=["csv"], key="pivot_file")
        pivot_time_col = st.text_input("时间列名称（例如：Time）", value="Time", key="pivot_time_col")
        pivot_data_col = st.text_input("数据列名称（例如：value）", value="value", key="pivot_data_col")
    
    with col2:
        def update_pivot_time_format():
            st.session_state['pivot_time_format'] = st.session_state['pivot_time_format_select']
        
        pivot_time_format = st.selectbox(
            "时间格式",
            options=common_formats,
            index=common_formats.index(st.session_state['pivot_time_format']),
            format_func=lambda x: f"{x} (例如：{datetime.now().strftime(x)})",
            key="pivot_time_format_select",
            on_change=update_pivot_time_format
        )
        pivot_granularity = st.selectbox("时间粒度", options=["15min", "1h"], index=0, key="pivot_granularity")
        pivot_agg_method = st.selectbox("聚合方法", options=["average", "max", "min"], index=0, key="pivot_agg_method")
    
    if st.button("生成透视表", key="generate_pivot"):
        if not pivot_file:
            st.error("请先上传一个 CSV 文件！")
        elif not pivot_time_col or not pivot_data_col:
            st.error("必须指定时间列和数据列名称！")
        else:
            with st.spinner("正在生成透视表..."):
                pivot_df = process_pivot(pivot_file, pivot_time_col, pivot_data_col, pivot_time_format, pivot_granularity, pivot_agg_method)
                if pivot_df is not None:
                    st.session_state['pivot_df'] = pivot_df
                    st.success("透视表生成成功！")
    
    if 'pivot_df' in st.session_state and st.session_state['pivot_df'] is not None:
        st.subheader("透视表结果")
        st.dataframe(st.session_state['pivot_df'], use_container_width=True)
    
        csv = st.session_state['pivot_df'].to_csv(encoding='utf-8-sig')
        st.download_button(
            label="保存透视表",
            data=csv,
            file_name=f"pivot_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="save_pivot"
        )

# 时间序列转换标签页
with tab2:
    st.header("时间序列转换")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unpivot_file = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx", "xls"], key="unpivot_file")
        unpivot_sheet_name = st.text_input("工作表名称（适用于 Excel 文件）", value="", key="unpivot_sheet_name")
    
    with col2:
        unpivot_header_cell = st.text_input("表头单元格（例如：A5）", value="A5", key="unpivot_header_cell")
        unpivot_granularity = st.selectbox("时间粒度", options=["15min", "1h"], index=0, key="unpivot_granularity")
    
    if st.button("转换为时间序列", key="generate_unpivot"):
        if not unpivot_file:
            st.error("请先上传一个文件！")
        elif not unpivot_header_cell:
            st.error("必须指定表头单元格！")
        elif not re.match(r'^[A-Z]+\d+$', unpivot_header_cell):
            st.error("表头单元格必须为类似 'A5' 的格式！")
        elif unpivot_file.name.endswith(('.xlsx', '.xls')) and not unpivot_sheet_name:
            st.error("Excel 文件必须指定工作表名称！")
        else:
            with st.spinner("正在转换为时间序列..."):
                file_type = 'excel' if unpivot_file.name.endswith(('.xlsx', '.xls')) else 'csv'
                unpivot_df = process_unpivot(unpivot_file, file_type, unpivot_sheet_name, unpivot_header_cell, unpivot_granularity)
                if unpivot_df is not None:
                    st.session_state['unpivot_df'] = unpivot_df
                    st.success("时间序列数据生成成功！")
    
    if 'unpivot_df' in st.session_state and st.session_state['unpivot_df'] is not None:
        st.subheader("时间序列结果")
        st.dataframe(st.session_state['unpivot_df'], use_container_width=True)
    
        csv = st.session_state['unpivot_df'].to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="保存时间序列",
            data=csv,
            file_name=f"time_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="save_unpivot"
        )

# 风电出力率分析标签页
with tab3:
    st.header("风电出力率分析")
    
    # 初始化 session state 以保存时间格式
    if 'wind_time_format' not in st.session_state:
        st.session_state['wind_time_format'] = common_formats[0]  # 默认格式
    
    col1, col2 = st.columns(2)
    
    with col1:
        wind_file = st.file_uploader("上传 CSV 文件", type=["csv"], key="wind_file")
        if wind_file:
            try:
                # 检测分隔符
                wind_file.seek(0)
                sample = wind_file.read(1024).decode('utf-8-sig', errors='ignore')
                wind_file.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                # 尝试多种编码
                encodings = ['utf-8-sig', 'gbk', 'utf-8']
                df = None
                for enc in encodings:
                    try:
                        wind_file.seek(0)
                        df = pd.read_csv(wind_file, encoding=enc, sep=delimiter)
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    raise ValueError("无法解析 CSV 文件，请检查文件编码")
                if df.empty:
                    raise ValueError("上传的 CSV 文件为空")
                columns = df.columns.tolist()
                st.write("检测到的列名：", columns)
                st.write("原始列名（检查编码问题）：", [repr(col) for col in df.columns])
                # 显示文件前几行
                if st.checkbox("显示 CSV 内容调试信息", key="wind_debug"):
                    wind_file.seek(0)
                    df_debug = pd.read_csv(wind_file, encoding=enc, sep=delimiter, nrows=5)
                    st.write("CSV 前5行：", df_debug)
                    if time_col in df_debug.columns:
                        st.write("时间列前5个值：", df_debug[time_col].head().tolist())
            except Exception as e:
                st.error(f"读取文件失败: {str(e)}")
                columns = []
        else:
            columns = []
        
        time_col = st.selectbox("时间列", columns, key="wind_time_col", help="选择时间戳列（例如：2024/01/01 0:00）")
        value_col = st.selectbox("值列（发电量）", columns, key="wind_value_col", help="选择发电量列")
        unit_col = st.selectbox("机组编号列", columns, key="wind_unit_col", help="选择机组编号列")
    
    with col2:
        def update_wind_time_format():
            st.session_state['wind_time_format'] = st.session_state['wind_time_format_select']
        
        time_format = st.selectbox(
            "时间格式",
            options=common_formats,
            index=common_formats.index(st.session_state['wind_time_format']),
            format_func=lambda x: f"{x} (例如：{datetime.now().strftime(x)})",
            key="wind_time_format_select",
            on_change=update_wind_time_format
        )
        value_unit = st.selectbox("值列单位", ["kWh", "MWh"], key="wind_value_unit", help="发电量数据的单位")
        total_capacity = st.number_input(
            f"总容量（{value_unit}）",
            min_value=0.0,
            value=22.719 if value_unit == "MWh" else 22719.0,
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
            wind_file.seek(0)
            sample = wind_file.read(1024).decode('utf-8-sig', errors='ignore')
            wind_file.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            encodings = ['utf-8-sig', 'gbk', 'utf-8']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(wind_file, encoding=enc, sep=delimiter)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise ValueError("无法解析 CSV 文件，请检查文件编码")
            if df.empty:
                raise ValueError("CSV 文件为空")
            if time_col not in df.columns:
                raise ValueError(f"时间列 '{time_col}' 不存在")
            df[time_col] = pd.to_datetime(df[time_col], format=time_format, errors='coerce')
            if df[time_col].isna().any():
                invalid_rows = df[df[time_col].isna()][time_col].head().tolist()
                raise ValueError(f"时间列包含无效格式，请检查时间格式 '{time_format}'。无效值示例: {invalid_rows}")
            # 按机组和时间排序，计算每个机组的时间差
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
    
    if st.button("运行分析", key="wind_analyze"):
        if not wind_file:
            st.error("请先上传一个 CSV 文件！")
        elif not time_col or not value_col or not unit_col:
            st.error("必须选择时间列、值列和机组编号列！")
        elif total_capacity <= 0:
            st.error("总容量必须大于0！")
        else:
            with st.spinner("正在分析数据..."):
                try:
                    wind_file.seek(0)
                    sample = wind_file.read(1024).decode('utf-8-sig', errors='ignore')
                    wind_file.seek(0)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    encodings = ['utf-8-sig', 'gbk', 'utf-8']
                    df = None
                    for enc in encodings:
                        try:
                            df = pd.read_csv(wind_file, encoding=enc, sep=delimiter)
                            break
                        except UnicodeDecodeError:
                            continue
                    if df is None:
                        raise ValueError("无法解析 CSV 文件，请检查文件编码")
                    if df.empty:
                        raise ValueError("CSV 文件为空")
                    if time_col not in df.columns:
                        raise ValueError(f"时间列 '{time_col}' 不存在")
                    if value_col not in df.columns:
                        raise ValueError(f"值列 '{value_col}' 不存在")
                    if unit_col not in df.columns:
                        raise ValueError(f"机组编号列 '{unit_col}' 不存在")
                    df[time_col] = pd.to_datetime(df[time_col], format=time_format, errors='coerce')
                    if df[time_col].isna().any():
                        invalid_rows = df[df[time_col].isna()][time_col].head().tolist()
                        raise ValueError(f"时间列包含无效格式，请检查时间格式 '{time_format}'。无效值示例: {invalid_rows}")
                    results_df = calculate_power_rate(
                        df, time_col, value_col, unit_col, value_unit, total_capacity, sample_period, percentile
                    )
                    st.session_state['wind_df'] = results_df
                    # 调试 CSV 输出
                    if st.checkbox("调试 CSV 输出内容", key="wind_csv_debug"):
                        csv_debug = '\ufeff' + results_df.to_csv(index=False, encoding='utf-8')
                        st.text("CSV 输出内容预览（前1000字符）：")
                        st.text(csv_debug[:1000])
                    st.success("分析完成！")
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")
    
    if 'wind_df' in st.session_state and st.session_state['wind_df'] is not None:
        st.subheader("分析结果")
        st.dataframe(
            st.session_state['wind_df'].style.format({f'{percentile}%概率出力率': '{:.4f}'}, na_rep='无有效数据'),
            use_container_width=True
        )
        csv = '\ufeff' + st.session_state['wind_df'].to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="保存结果",
            data=csv,
            file_name=f"wind_power_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="save_wind"
        )

# 关于标签页
with tab4:
    st.header("关于 (About)")
    st.markdown("""
    ### 工具简介
    时间序列数据转换与分析工具是为电力数据分析设计的多功能应用，支持以下三种功能：

    #### 1. 透视表生成
    **功能**：将时间序列数据转换为透视表，按指定时间粒度（15分钟或1小时）对数据进行聚合（平均、最大、最小值），支持多种时间格式选择。  
    **使用方法**：
    - 上传一个 CSV 文件，包含时间列和数据列（例如：时间戳和发电量）。
    - 输入时间列名称（例如：Time）和数据列名称（例如：value）。
    - 选择时间格式（例如：%Y/%m/%d %H:%M:%S）。
    - 选择时间粒度（15分钟或1小时）和聚合方法（平均、最大、最小）。
    - 点击“生成透视表”按钮，查看结果并可下载为 CSV 文件。

    #### 2. 时间序列转换
    **功能**：将透视表格式的数据转换为时间序列格式，支持 CSV 和 Excel 文件输入。  
    **使用方法**：
    - 上传 CSV 或 Excel 文件，文件需包含日期索引和时间列（如 00:00, 01:00 等）。
    - 对于 Excel 文件，指定工作表名称。
    - 输入表头单元格（例如：A5），表示数据区域的起始位置。
    - 选择时间粒度（15分钟或1小时）。
    - 点击“转换为时间序列”按钮，查看结果并可下载为 CSV 文件。

    #### 3. 风电出力率分析
    **功能**：分析风电数据，计算各季节（春季（3-5月）、夏季（6-8月）、秋季（9-11月）、冬季（12月，1-2月）、全年）在特定时段（19:00-22:00、11:00-14:00、1:00-4:00）的指定分位出力率，支持多机组数据和不同采样周期（1小时、30分钟、15分钟）。  
    **使用方法**：
    - 上传 CSV 文件，包含机组编号、时间戳和发电量数据（例如：机组、日期时间、风电出力）。
    - 选择时间列（例如：日期时间）、值列（例如：风电出力）和机组编号列（例如：机组）。
    - 选择时间格式（例如：%Y/%m/%d %H:%M）。
    - 选择值列单位（kWh 或 MWh）、总容量、采样周期和分位数（例如：5 表示前5%分位）。
    - 点击“运行分析”按钮，查看结果并可下载为 CSV 文件。

    ### 研发人员信息
    **姓名**：倪程捷  
    **职称**：高级工程师  
    **单位**：中国电力工程顾问集团华东电力设计院  
    **邮箱**：chengjie_ni@163.com
    """)
