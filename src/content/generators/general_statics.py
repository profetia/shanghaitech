import pygal
from pygal.style import Style
from pygal.style import lighten
from pygal.style import darken
from copy import deepcopy
import pandas as pd
import numpy as np

savepath = "./static/general_statics.svg"

def draw(data):
    conf = pygal.Config()
    conf.style = Style(
        background = "#ebebeb", 
        plot_background = "#ebebeb", 
        foreground = 'rgba(0, 0, 0, 0.9)', 
        foreground_strong = 'rgba(0, 0, 0, 0.9)', 
        foreground_subtle = 'rgba(0, 0, 0, 0.5)', 
        opacity = '.6', 
        opacity_hover = '.9', 
        major_label_font_size = 54, 
        label_font_size = 54, 
        tooltip_font_size = 70, 
        no_data_font_size = 150, 
        colors = ("#5b69ab", "#c8435c", '#39929a',
        lighten('#d94e4c', 10), darken('#39929a', 15), lighten('#e5884f', 17),
        darken('#d94e4c', 10), '#234547')
    )
    conf.height = 1000
    conf.width = 3000
    conf.fill = True
    conf.x_labels = list(data['date'])
    conf.x_labels_major = [data.loc[i, 'date'] for i in np.linspace(30, len(data) - 30, 5, dtype = int)]
    conf.show_minor_x_labels = False
    conf.truncate_label = -1
    conf.show_y_labels = False
    conf.value_formatter = lambda x:"{:,}".format(int(x))
    conf.legend_at_bottom = True
    conf.show_legend = False
    conf.margin = 0
    chart = pygal.Line(conf)
    chart.add("Confirmed Deaths", data['deaths'])
    chart.add("Confirmed Cases", data["cases"], secondary = True)
    chart.render_to_file(savepath)

def general_statics(raw_data):
    data = deepcopy(raw_data)
    for i in np.linspace(30, len(data) - 30, 5, dtype = int):
        data.loc[i, 'date'] = data.loc[i, 'date'][:7]
    draw(data)