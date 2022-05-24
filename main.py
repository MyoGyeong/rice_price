import pandas as pd
import streamlit as st
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from fbprophet import Prophet
st.title("쌀 값 예측")
st.subheader("쌀 값을 예측합니다.")
st.text("안녕하세요! 쌀값을 알고 싶은 지역을 선택하시면, 예측된 쌀값 그래프가 나타납니다. ")
st.sidebar.header("Menu")
select_city = st.sidebar.selectbox('쌀 값을 알고싶은 도시를 고르세요.', [
    '서울','대전','대구','부산','광주'
])
st.sidebar.write("선택하신 도시는 "+select_city+"입니다.")
year = st.sidebar.slider("Year", 1,5)
period = year*365
st.sidebar.write(str(year)+"년 뒤를 예측합니다.")
if select_city == '서울':
    data = pd.read_csv("C:\\Users\\thsay\\PycharmProjects\\pythonProject2\\data\\seoul.csv")
elif select_city =='광주':
    data = pd.read_csv("C:\\Users\\thsay\\PycharmProjects\\pythonProject2\\data\\gwangju.csv")
elif select_city=='대구':
    data = pd.read_csv("C:\\Users\\thsay\\PycharmProjects\\pythonProject2\\data\\daegu.csv")
elif select_city=='대전':
    data = pd.read_csv("C:\\Users\\thsay\\PycharmProjects\\pythonProject2\\data\\daejeon.csv")
else :
    data = pd.read_csv("C:\\Users\\thsay\\PycharmProjects\\pythonProject2\\data\\busan.csv")


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['price'], name="price"))
    fig.layout.update(title_text = "time seriese data with rangeslider", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

df_train = data[['date','price']]
df_train = df_train.rename(columns={
    "date":"ds",
    "price":"y"
})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.write(f'Forecast plot for a years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
