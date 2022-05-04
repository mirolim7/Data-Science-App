import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import streamlit_theme as stt
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import expon
from scipy.stats import uniform
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


sns.set_style('whitegrid')

stt.set_theme({'primary': '#1b3388'})

st.title('Akzhol Sovet, Mirolim Saidakhmatov, BDA-1902, Final Project')

st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.title('SideBar')

st.sidebar.header('The 1st task')
ch1 = st.sidebar.checkbox('cumulative from mean')
ch2 = st.sidebar.checkbox('cumulative')
ch3 = st.sidebar.checkbox('complementary cumulative')
#btn1 = st.sidebar.button('Show')

def npd(x):
    cnst = 1.0 / np.sqrt(2 * np.pi)
    return (cnst * np.exp((-x ** 2) / 2.0))

snt = pd.DataFrame(data=[], index=np.round(np.arange(0, 3.5, .1), 2),
                   columns=np.round(np.arange(0.00, .1, .01), 2))

for i in snt.index:
    for col in snt.columns:
        z = np.round(i + col, 2)
        value, _ = quad(npd, np.NINF, z)
        snt.loc[i, col] = value

if ch1 or ch2 or ch3:
    st.header('1st Task:')

if ch1:
    st.write('Cumulative from mean')
    @st.cache(persist=True)
    def load():
        data = pd.read_excel('cfm.xlsx', index_col=0)
        return data


    data = load()
    st.write(data)

if ch2:
    st.write('Cumulative')
    st.dataframe(snt)

if ch3:
    st.write('Complementary cumulative')
    @st.cache(persist=True)
    def load():
        data = pd.read_excel('cp.xlsx', index_col=0)
        return data


    data = load()
    st.write(data)

st.sidebar.header('The 2nd task')

option = st.sidebar.selectbox('Select distribution', ('Normal', 'Poisson', 'Gamma', 'Exponential', 'Uniform'))
st.header('2nd Task:')
if option == 'Normal':
    data = np.arange(-3.5, 3.5, 0.1)
    pdf = norm.pdf(data, loc=0, scale=1)
    fig, ax = plt.subplots()
    sns.lineplot(data, pdf, ax=ax, color='black')
    plt.title('Normal Distribution')
    st.pyplot(fig)
    st.write('A normal distribution (aka a Gaussian distribution) is a continuous probability distribution '
             'for real-valued variables. It is a symmetric distribution where most of the observations '
             'cluster around a central peak, which we call the mean.')

if option == 'Poisson':
    data_poisson = poisson.rvs(mu=3, size=10000)
    fig, ax = plt.subplots()
    ax = sns.distplot(data_poisson, bins=30, kde=False)
    ax.set(xlabel='Poisson Distribution', ylabel='Frequency')
    st.pyplot(fig)
    st.write('Poisson random variable is typically used to model the number of times an event happened in a time interval. '
        'For example, the number of users visited on a website in an interval can be thought of a Poisson process.')

if option == 'Gamma':
    data_gamma = gamma.rvs(a=5, size=10000)
    fig, ax = plt.subplots()
    ax = sns.distplot(data_gamma, bins=100)
    ax.set(xlabel='Gamma Distribution', ylabel='Frequency')
    st.pyplot(fig)
    st.write('The gamma distribution is a two-parameter family of continuous probability distributions. While it is used rarely'
             ' in its raw form but other popularly used distributions like exponential, chi-squared, erlang distributions are '
             'special cases of the gamma distribution.')


if option == 'Exponential':
    data_expon = expon.rvs(scale=1, loc=0, size=1000)
    fig, ax = plt.subplots()
    ax = sns.distplot(data_expon, bins=100)
    ax.set(xlabel='Exponential Distribution', ylabel='Frequency')
    st.pyplot(fig)
    st.write('The exponential distribution describes the time between events in a Poisson point process, i.e., a process in '
             'which events occur continuously and independently at a constant average rate.')


if option == 'Uniform':
    n = 10000
    start = 10
    width = 20
    data_uniform = uniform.rvs(size=n, loc=start, scale=width)
    fig, ax = plt.subplots()
    ax = sns.distplot(data_uniform, bins=100)
    ax.set(xlabel='Uniform Distribution ', ylabel='Frequency')
    st.pyplot(fig)
    st.write('Perhaps one of the simplest and useful distribution is the uniform distribution. The probability distribution function of '
             'the continuous uniform distribution is: 𝑓(𝑥)={1𝑏−𝑎𝑓𝑜𝑟𝑎≤𝑥≤𝑏,0𝑓𝑜𝑟𝑥<𝑎𝑜𝑟𝑥>𝑏 Since any interval of numbers of equal width has an '
             'equal probability of being observed, the curve describing the distribution is a rectangle, with constant height across the '
             'interval and 0 height elsewhere. Since the area under the curve must be equal to 1, the length of the interval determines '
             'the height of the curve.')


st.sidebar.header('The 3-4-5-6th tasks')

m3 = st.sidebar.number_input('Input mean:')
st3 = st.sidebar.number_input('Input std:')
x1 = st.sidebar.number_input('Input x1:')
x2 = st.sidebar.number_input('Input x2:')
btn3 = st.sidebar.button('Plot')
if btn3:
    if x1 and not x2:
        st.header('4th Task:')
        data = np.arange(m3-3*st3, m3+3*st3, 0.01)
        pdf = norm.pdf(data, loc=m3, scale=st3)
        fig, ax = plt.subplots()
        pX = np.arange(m3-3*st3, x1, 0.01)  # X values for fill
        pY = norm.pdf(pX, loc=m3, scale=st3)  # Y values for fill
        plt.fill_between(pX, pY, alpha=0.5)
        pro = round(norm.cdf(x=x1, loc=m3, scale=st3), 2)
        plt.text(m3-1.5*st3, 0.02, pro, fontsize=20)  # Add text at position
        sns.lineplot(data, pdf, ax=ax, color='black')
        plt.title('Normal Dist. with mean = {} and std_dv = {}'.format(m3, st3))
        st.pyplot(fig)

    if x2 and not x1:
        st.header('5th Task:')
        data = np.arange(m3 - 3 * st3, m3 + 3 * st3, 0.01)
        pdf = norm.pdf(data, loc=m3, scale=st3)
        fig, ax = plt.subplots()
        pX = np.arange(x2, m3 + 3 * st3, 0.01)  # X values for fill
        pY = norm.pdf(pX, loc=m3, scale=st3)  # Y values for fill
        plt.fill_between(pX, pY, alpha=0.5)
        pro = round(norm.sf(x=x2, loc=m3, scale=st3), 2)
        plt.text(m3 + 0.5 * st3, 0.02, pro, fontsize=20)  # Add text at position
        sns.lineplot(data, pdf, ax=ax, color='black')
        plt.title('Normal Dist. with mean = {} and std_dv = {}'.format(m3, st3))
        st.pyplot(fig)

    if x1 and x2:
        st.header('6th Task:')
        data = np.arange(m3 - 3 * st3, m3 + 3 * st3, 0.01)
        pdf = norm.pdf(data, loc=m3, scale=st3)
        fig, ax = plt.subplots()
        pX = np.arange(x1, x2, 0.01)  # X values for fill
        pY = norm.pdf(pX, loc=m3, scale=st3)  # Y values for fill
        plt.fill_between(pX, pY, alpha=0.5)
        pro = round((norm.cdf(x=x2, loc=m3, scale=st3) - norm.cdf(x=x1, loc=m3, scale=st3)), 2)
        plt.text(m3 + 0.5 * st3, 0.02, pro, fontsize=20)  # Add text at position
        sns.lineplot(data, pdf, ax=ax, color='black')
        plt.title('Normal Dist. with mean = {} and std_dv = {}'.format(m3, st3))
        st.pyplot(fig)


st.sidebar.header('The 7th task')

collect_numbers = lambda x : [int(i) for i in re.split("[^0-9]", x) if i != ""]

xlist = st.sidebar.text_input("Enter values for X: ")
ylist = st.sidebar.text_input("Enter values for Y: ")
btn7 = st.sidebar.button('Plot Scatter plot')
st.sidebar.write("---For convenience---")
st.sidebar.write("90, 100, 90, 80, 87, 75")
st.sidebar.write("950, 1100, 850, 750, 950, 775")

lm = LinearRegression()
if btn7:
    x7 = collect_numbers(xlist)
    y7 = collect_numbers(ylist)

    st.header('7th Task:')
    fig, ax = plt.subplots()
    sns.regplot(x7, y7, ax=ax)
    x = np.array(x7).reshape(-1, 1)
    y = np.array(y7)
    lm.fit(x, y)
    st.pyplot(fig)
    st.write('Coefficent of determination is: ', lm.score(x, y))
    st.write('Intercept is: ', lm.intercept_)
    st.write('Slope is: ', lm.coef_)


#np.random.seed(0)
#x = np.random.randn(100)
#fig, ax = plt.subplots()
#sns.distplot(x, fit=norm, hist=False, ax=ax, kde=False)
#st.pyplot(fig)

