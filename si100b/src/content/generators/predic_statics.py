import math
import pygal
from pygal.style import Style
from pygal.style import lighten
from pygal.style import darken
from datetime import datetime
from datetime import timedelta
from copy import deepcopy
import pandas as pd
import numpy as np

savepath = "./static"

configue = {
    "cases":  
        {
            "color": "#c8435c", 
            "title": "Predicted Cases in 7 Days", 
            "legend": "Cumulative Cases", 
            "savepath": "predic_cases"
        },
    "deaths": 
        {
            "color": "#5b69ab", 
            "title": "Predicted Cases in 7 Days", 
            "legend": "Cumulative Deaths", 
            "savepath": "predic_deaths"
        },
}

def FIT(fitx, fity, num = 5):
    fitx = np.array(fitx)
    fity = np.array(fity)
    fit = np.polyfit(fitx,fity,num)
    return np.poly1d(fit)

def predict(data, name):
    y = []
    for i in range(len(data) - 99, len(data),7):
        y.append(data.loc[i, name] - data.loc[i - 7, name])
    x = list(range(1, 16))
    P1 = FIT(x,y)
    u = 0
    sig = math.sqrt(0.2)
    x = np.linspace(u - 3 * sig, u + 3 * sig, 13)
    y_sig = list(np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig))
    y_sig = y_sig[3:10]
    data1 = [[i] for i in range(7)]
    for i in range(7):
        data1[i].append(y_sig[i] * P1(16) / sum(y_sig))
    predict = []
    o = datetime.strptime(data.loc[len(data) - 1]['date'],'%Y-%m-%d').weekday()
    for i in range(7):
        predict.append(int(data1[o][1]))
        o = (o + 1) % 7
    return predict

def get_data(data):
    date = []
    new_death = []
    new_case = []
    length = len(data)
    for i in range(length - 7, length):
        date.append(data.loc[i, 'date'])
        new_case.append(data.loc[i, 'cases'] - data.loc[length - 8, 'cases'])
        new_death.append(data.loc[i, 'deaths'] - data.loc[length - 8, 'deaths'])
    stdtime = datetime.strptime(date[-1], "%Y-%m-%d")
    for i in range(7):
        future = stdtime + timedelta(days = i + 1)
        date.append(future.strftime("%Y-%m-%d"))
    date = list(map(lambda x: x[5:], date))
    predic_case = predict(data, 'cases')
    predic_death = predict(data, 'deaths')
    cases = []
    deaths = []
    predic_case[0] += new_case[-1]
    predic_death[0] += new_death[-1]
    for i in range(1, 7):
        predic_case[i] = predic_case[i - 1] + predic_case[i]
        predic_death[i] = predic_death[i - 1] + predic_death[i]
    for i in range(7):
        cases.append(
            {"value": new_case[i]}
        )
        deaths.append(
            {"value": new_death[i]}
        )
    for i in range(7):
        cases.append(
            {
                "value": predic_case[i], 
                'style': 'fill: #ebebeb; stroke: black; stroke-dasharray: 15, 10, 5, 10, 15'
            }
        )
        deaths.append(
            {
                "value": predic_death[i], 
                'style': 'fill: #ebebeb; stroke: black; stroke-dasharray: 15, 10, 5, 10, 15'
            }
        )
    return {'date': date, 'cases': cases, 'deaths': deaths}

def draw(name, data):
    setting = configue[name]
    conf = pygal.Config()
    conf.style = Style(
        background = "rgba(245, 245, 245, 0.3)", 
        plot_background = "rgba(245, 245, 245, 0.3)", 
        foreground = 'rgba(0, 0, 0, 0.9)', 
        foreground_strong = 'rgba(0, 0, 0, 0.9)', 
        foreground_subtle = 'rgba(0, 0, 0, 0.5)', 
        opacity = '.6', 
        opacity_hover = '.9', 
        label_font_size = 63, 
        value_font_size = 63, 
        tooltip_font_size = 81, 
        no_data_font_size = 150, 
        colors = (setting['color'], '#e5884f', '#39929a',
        lighten('#d94e4c', 10), darken('#39929a', 15), lighten('#e5884f', 17),
        darken('#d94e4c', 10), '#234547')
    )
    conf.height = 1200
    conf.width = 4000
    conf.x_labels = data['date']
    conf.truncate_label = -1
    conf.show_y_labels = False
    conf.value_formatter = lambda x:"{:,}".format(int(x))
    conf.legend_at_bottom = True
    conf.show_legend = False
    conf.margin = 0
    chart = pygal.Bar(conf)
    chart.add(setting['legend'], data[name])
    chart.render_to_file(f"{savepath}/{setting['savepath']}.svg")
    

def predic_statics(raw_data):
    data = deepcopy(raw_data)
    data = get_data(data)
    draw("cases", data)
    draw("deaths", data)
    return {
        "prediccases": "{:,}".format(data['cases'][-1]["value"] - data['cases'][-8]["value"]), 
        "predicdeaths": "{:,}".format(data["deaths"][-1]["value"] - data["deaths"][-8]["value"])
        }