import pygal
from pygal.style import Style
from pygal.style import lighten
from pygal.style import darken
from copy import deepcopy
import pandas as pd

savepath = "./static"

configue = {
    "cases":  
        {
            "color": "#c8435c", 
            "title": "New Cases in 14 Days", 
            "legend": "New Cases", 
            "savepath": "d14_cases"
        },
    "deaths": 
        {
            "color": "#5b69ab", 
            "title": "New Deaths in 14 Days", 
            "legend": "New Deaths", 
            "savepath": "d14_deaths"
        },
}

def get_data(data):
    date = []
    new_case = []
    new_death = []
    for i in range(len(data) - 14, len(data)):
        date.append(data.loc[i, 'date'][5:])
        new_case.append(data.loc[i, 'cases'] - data.loc[i - 1, 'cases'])
        new_death.append(data.loc[i, 'deaths'] - data.loc[i - 1, 'deaths'])
    return {'date': date, 'cases': new_case, 'deaths': new_death}

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
    

def d14_statics(raw_data):
    data = deepcopy(raw_data)
    data = get_data(data)
    draw("cases", data)
    draw("deaths", data)
    return {
        "d14cases": "{:,}".format(sum(data['cases'])), 
        "d14deaths": "{:,}".format(sum(data["deaths"]))
        }