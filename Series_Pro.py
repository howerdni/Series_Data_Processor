import streamlit as st
import pandas as pd

class PivotDataProcessor:
    def __init__(self):
        self.df = None

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

    def pivot_data(self, row_column, column_column, value_column):
        if self.df is not None:
            try:
                pivot_table = self.df.pivot_table(index=row_column, columns=column_column, values=value_column, aggfunc='sum')
                return pivot_table
            except Exception as e:
                st.error(f"生成数据透视表时出错: {str(e)}")
                return None
        return None

def main():
    st.title("数据透视表生成器")
    processor = PivotDataProcessor()

    st.header("数据上传与透视表生成")
    uploaded_file = st.file_uploader("上传Excel或CSV文件", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        if processor.load_data(uploaded_file):
            st.success("文件上传成功！")
            columns = processor.df.columns.tolist()
            
            row_column = st.selectbox("选择行字段", columns, key="row_column_pivot")
            column_column = st.selectbox("选择列字段", columns, key="column_column_pivot")
            value_column = st.selectbox("选择值字段", columns, key="value_column_pivot")
            
            if st.button("生成数据透视表"):
                pivot_table = processor.pivot_data(row_column, column_column, value_column)
                if pivot_table is not None:
                    st.write("数据透视表：")
                    st.dataframe(pivot_table)
                    # 提供下载透视表的功能
                    csv = pivot_table.to_csv()
                    st.download_button(
                        label="下载透视表为CSV",
                        data=csv,
                        file_name="pivot_table.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
