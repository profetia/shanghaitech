import pandas as pd
from .state_draw import draw_d14
from .state_draw import draw_general

dailypath = "./content/templates/state_data.csv"

def readable(x):
    return "{:,}".format(int(x))

def percent(x):
    if x <= 1:
        return "{:.2%}".format(x)
    else:
        return "{:.2f}".format(x) + "%"

def daily(name):
    date = [] 
    new_cases = []
    new_deaths = []
    df = pd.read_csv(dailypath).groupby("state")
    state_data = df.get_group(name)
    for i in range(len(state_data) - 14, len(state_data)):
        date.append(state_data.iloc[i]['date'][5:])
        new_cases.append(state_data.iloc[i]['cases'] - state_data.iloc[i - 1]['cases'])
        new_deaths.append(state_data.iloc[i]['deaths'] - state_data.iloc[i - 1]['deaths'])
    data = {
        "date": list(state_data["date"]), 
        "cases": list(state_data["cases"]), 
        "deaths": list(state_data["deaths"]), 
        "d14_date": date, 
        "new_cases": new_cases, 
        "new_deaths": new_deaths
    }
    draw_d14("new_cases", data)
    draw_d14("new_deaths", data)
    draw_general(data)
    return {
        "cases": readable(state_data.iloc[-1]["cases"]), 
        "deaths": readable(state_data.iloc[-1]["deaths"]), 
        "death_rate": percent(state_data.iloc[-1]["deaths"] / state_data.iloc[-1]["cases"]), 
        "new_cases": readable(new_cases[-1]), 
    }

from .state_draw import draw_vac

vacpath = "./content/templates/state_vac.csv"

def vaccine(name):
    df = pd.read_csv(vacpath)
    state_vac = df.groupby("Province_State").get_group(name)
    vacs = {
        "full_vac": state_vac.iloc[0]['People_Fully_Vaccinated'], 
        "d1_vac": state_vac.iloc[0]['People_Partially_Vaccinated'] + state_vac.iloc[0]['People_Fully_Vaccinated'], 
    }
    draw_vac(vacs)

def load(name, path):
    df = pd.read_csv(path)
    df = df.groupby('NAME').get_group(name)
    if 'state' in df.columns:
        df = df.drop(['state'], axis = 1)
    return df.iloc[0]

inspath = "./content/templates/state_ins_.csv"

def cook(data):
    re = list(data)
    re = re[1:]
    re = list(map(lambda x: int(x) if x != None else 0, re))
    return re

from .state_draw import draw_ins

def insurance(name):
    data = {
        '<19': cook(load(name, inspath[:-4] + '19' + inspath[-4:])), 
        '19-65': cook(load(name, inspath[:-4] + '19-65' + inspath[-4:])), 
        '>65': cook(load(name, inspath[:-4] + '65' + inspath[-4:]))
    }
    pop = {
        '19': sum(data['<19']), 
        '19-65': sum(data['19-65']), 
        '65': sum(data['>65'])
    }
    pop_all = sum(list(pop.values()))
    for key in data.keys():
        data[key] = list(map(lambda x: x / pop_all, data[key]))
    draw_ins(data)

from .state_draw import draw_pop

poppath = "./content/templates/state_pop.csv"
povpath = "./content/templates/state_pov.csv"

ref = {
    "_race": {
        'white': 'White', 'black': 'Black', 'asia': "Asian",
        'amindia': 'American Indian', "pacific": "Pacific Islander", 
        'other': "Other Races", "2m": "Two or more" 
    }, 
    "_anc": {
        'nonhispain': "Non-Hispainic", "hispain": "Hispanic or Latino"
    },
    '_age': {
        '<15': "Under 15", "15-24": "Age 15-24", "25-34": "Age 25-34", 
        '35-64': "Age 35-64", "65-74": "Age 65-74", ">75": "Over 75"
    }
}

def render(name, kind):
    pop_race = load(name, poppath[:-4] + kind + poppath[-4:])
    pop_race = dict(pop_race)
    data = {}
    for key, value in pop_race.items():
        if key != 'NAME':
            new_key = ref[kind][key]
            data[new_key] = value
    draw_pop(data, kind)
    if kind == '_age': 
        return {'gt65': readable(data['Age 65-74'] + data['Over 75'])}

def population(name):
    re = {}
    pop_all = load(name, poppath)
    re['pop_all'] = readable(pop_all['pop'])
    pov = load(name, povpath)
    re['pov'] = readable(pov['poverty'])
    pop_all = load(name, poppath)
    pop_all = pop_all['pop']
    render(name, '_race')
    render(name, '_anc')
    re.update(render(name, '_age'))
    return re

def state_statics(name):
    statics = daily(name)
    vaccine(name)
    insurance(name)
    statics.update(population(name))
    return statics