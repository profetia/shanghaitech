import pygal
from pygal import colors
from pygal.style import Style
from pygal.style import lighten
from pygal.style import darken
import numpy as np

savepath = "./static"

configue = {
    "new_cases":  
        {
            "color": "#c8435c", 
            "title": "NEW CASES IN 14 DAYS", 
            "legend": "New Cases", 
            "savepath": "county_cases"
        },
    "new_deaths": 
        {
            "color": "#5b69ab", 
            "title": "NEW DEATHS IN 14 DAYS", 
            "legend": "New Deaths", 
            "savepath": "county_deaths"
        },
}

def cov(x):
    try:
        x = "{:,}".format(int(x))
    except Exception:
        x = None
    return x

def draw_d14(name, data):
    setting = configue[name]
    conf = pygal.Config()
    conf.style = Style(
        background = "transparent", 
        plot_background = "transparent", 
        foreground = 'rgba(0, 0, 0, 0.9)', 
        foreground_strong = 'rgba(0, 0, 0, 0.9)', 
        foreground_subtle = 'rgba(0, 0, 0, 0.5)', 
        opacity = '.6', 
        opacity_hover = '.9', 
        title_font_size = 72, 
        label_font_size = 72, 
        major_label_font_size = 72, 
        value_font_size = 72, 
        tooltip_font_size = 90, 
        no_data_font_size = 150,
        colors = (setting['color'], '#e5884f', '#39929a',
        lighten('#d94e4c', 10), darken('#39929a', 15), lighten('#e5884f', 17),
        darken('#d94e4c', 10), '#234547')
    )
    conf.height = 1200
    conf.width = 4000
    conf.x_labels = data['d14_date']
    conf.truncate_label = -1
    conf.show_y_labels = False
    conf.value_formatter = cov
    conf.legend_at_bottom = True
    conf.show_legend = False
    conf.margin = 0
    chart = pygal.Bar(conf)
    chart.add(setting['legend'], data[name])
    chart.render_to_file(f"{savepath}/{setting['savepath']}.svg")

def draw_general(data):
    conf = pygal.Config()
    conf.style = Style(
        background = "transparent", 
        plot_background = "transparent", 
        foreground = 'rgba(0, 0, 0, 0.9)', 
        foreground_strong = 'rgba(0, 0, 0, 0.9)', 
        foreground_subtle = 'rgba(0, 0, 0, 0.5)', 
        opacity = '.6', 
        opacity_hover = '.9', 
        major_label_font_size = 72, 
        label_font_size = 72, 
        tooltip_font_size = 90, 
        no_data_font_size = 150, 
        colors = ("#5b69ab", "#c8435c", '#39929a',
        lighten('#d94e4c', 10), darken('#39929a', 15), lighten('#e5884f', 17),
        darken('#d94e4c', 10), '#234547')
    )
    conf.height = 1200
    conf.width = 4000
    conf.fill = True
    conf.x_labels = data['date']
    conf.x_labels_major = [data['date'][i] for i in np.linspace(30, len(data['date']) - 30, 5, dtype = int)]
    conf.show_minor_x_labels = False
    conf.truncate_label = -1
    conf.show_y_labels = False
    conf.value_formatter = cov
    conf.legend_at_bottom = True
    conf.show_legend = False
    conf.margin = 0
    chart = pygal.Line(conf)
    chart.add("Confirmed Deaths", data['deaths'])
    chart.add("Confirmed Cases", data["cases"], secondary = True)
    chart.render_to_file(f"{savepath}/county_general.svg")

def draw_vac(data, name):
    conf = pygal.Config()
    conf.style = Style(
        background = "transparent", 
        plot_background = "transparent", 
        foreground = 'rgba(0, 0, 0, 0.9)', 
        foreground_strong = 'rgba(0, 0, 0, 0.9)', 
        foreground_subtle = 'rgba(0, 0, 0, 0.5)', 
        opacity = '.6', 
        opacity_hover = '.9', 
        title_font_size = 81, 
        label_font_size = 63, 
        major_label_font_size = 63, 
        value_font_size = 63, 
        tooltip_font_size = 81, 
        no_data_font_size = 150,
        legend_font_size = 63, 
        colors = ("rgb(255, 158, 27)", 'rgb(207, 69, 32)', 'rgb(118, 35, 47)',  
        lighten('#d94e4c', 10), darken('#39929a', 15), lighten('#e5884f', 17), 
        darken('#d94e4c', 10), '#234547')
    )
    conf.truncate_label = -1
    conf.value_formatter = cov
    conf.legend_box_size = 63
    conf.margin = 0
    conf.inner_radius = .6
    conf.show_legend = False
    chart = pygal.Pie(conf)
    chart.add("Age 12-17", [data[f'{name}_12p']])
    chart.add("Age 18-64", [data[f'{name}_18p']])
    chart.add("Over 65", [data[f'{name}_65p']])
    chart.render_to_file(f"{savepath}/county_{name}.svg")

def draw_ins(data):
    conf = pygal.Config()
    conf.style = Style(
        background = "transparent", 
        plot_background = "transparent", 
        foreground = 'rgba(0, 0, 0, 0.9)', 
        foreground_strong = 'rgba(0, 0, 0, 0.9)', 
        foreground_subtle = 'rgba(0, 0, 0, 0.5)', 
        opacity = '.6', 
        opacity_hover = '.9', 
        title_font_size = 90, 
        label_font_size = 72, 
        major_label_font_size = 72, 
        value_font_size = 72, 
        tooltip_font_size = 90, 
        no_data_font_size = 150,
        legend_font_size = 72, 
        colors = ("rgb(255, 158, 27)", 'rgb(207, 69, 32)', 'rgb(118, 35, 47)',
        lighten('#d94e4c', 10), darken('#39929a', 15), lighten('#e5884f', 17),
        darken('#d94e4c', 10), '#234547')
    )
    conf.height = 4000
    conf.width = 3200
    conf.x_labels = ["  Employer Ins", "  Direct Purchase", "  Medicare", 
        "  Medicaid", "  VA Health Care", "  No Insurance"]
    conf.x_label_rotation = -40
    conf.truncate_label = -1
    conf.value_formatter = lambda x:"{:.2%}".format(x) if x != None else None
    conf.legend_box_size = 81
    conf.legend_at_bottom = True
    conf.legend_at_bottom_columns = 3
    conf.margin = 0
    chart = pygal.StackedBar(conf)
    chart.add('Under 19', data['<19'])
    chart.add('Age 19-65', data['19-65'])
    chart.add('Over 65', data['>65'])
    chart.render_to_file(f"{savepath}/county_ins.svg")

mapping = {
    '_race': 
        {
            'colors': ("rgb(0, 45, 114)", 'rgb(113, 153, 73)', 'rgb(161, 146, 178)',
                "rgb(134, 200, 188)", "rgb(207, 69, 32)", "rgb(255, 158, 27)", "rgb(118, 35, 47)"), 
            'name': 'Race'
        }, 
    '_anc':
        {
            'colors': ("rgb(255, 158, 27)", "rgb(134, 200, 188)", "rgb(0, 45, 114)", 
                'rgb(113, 153, 73)', 'rgb(161, 146, 178)', "rgb(207, 69, 32)", "rgb(118, 35, 47)"), 
            'name': 'Ethnicity'
        }, 
    '_age':
        {
            'colors': ("rgb(0, 45, 114)", 'rgb(113, 153, 73)', "rgb(134, 200, 188)", 
                "rgb(241, 196, 0)", "rgb(255, 158, 27)", "rgb(207, 69, 32)"), 
            'name': "Age Groups"
        }
}

def ylabel(data):
    tot = 0
    for val in data.values():
        tot += int(val)
    sz = int(np.log10(tot))
    tot = (tot // (10 ** (sz - 1))) * (10**(sz - 1))
    re = list(np.linspace(0, tot, 3))
    return re

def draw_pop(data, kind):
    setting = mapping[kind]
    conf = pygal.Config()
    conf.style = Style(
        background = "transparent", 
        plot_background = "transparent", 
        foreground = 'rgba(0, 0, 0, 0.9)', 
        foreground_strong = 'rgba(0, 0, 0, 0.9)', 
        foreground_subtle = 'rgba(0, 0, 0, 0.5)', 
        opacity = '.6', 
        opacity_hover = '.9', 
        title_font_size = 81, 
        label_font_size = 63, 
        major_label_font_size = 63, 
        value_font_size = 63, 
        tooltip_font_size = 81, 
        no_data_font_size = 150,
        legend_font_size = 63, 
        colors = setting['colors']
    )
    conf.height = 2000
    conf.width = 900
    conf.x_labels = [setting['name']]
    conf.y_labels = ylabel(data)
    conf.truncate_label = -1
    conf.value_formatter = cov
    conf.legend_box_size = 72
    conf.legend_at_bottom = True
    conf.legend_at_bottom_columns = 1
    conf.show_legend = False
    conf.truncate_legend = -1
    conf.margin = 0
    if 10000000 <= conf.y_labels[-1]:
        conf.margin_left = 75
    elif 1000000 <= conf.y_labels[-1] < 10000000:
        conf.margin_left = 63 * 3
    elif 100000 <= conf.y_labels[-1] < 1000000:
        conf.margin_left = 190
    elif 10000 <= conf.y_labels[-1] < 100000:
        conf.margin_left = 230
    elif 1000 <= conf.y_labels[-1] < 10000:
        conf.margin_left = 262
    else:
        conf.margin_left = 340
    chart = pygal.StackedBar(conf)
    for key, value in data.items():
        chart.add(key, [value])
    chart.render_to_file(f"{savepath}/county_pop{kind}.svg")