import pandas as pd
from copy import deepcopy

mapping = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AS": "American Samoa",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District Of Columbia",
    "FM": "Federated States Of Micronesia",
    "FL": "Florida",
    "GA": "Georgia",
    "GU": "Guam",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MH": "Marshall Islands",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "MP": "Northern Mariana Islands",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PW": "Palau",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VI": "Virgin Islands",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming"
}

def full_state(name):
    return mapping[name]

from .county_draw import draw_d14
from .county_draw import draw_general

dailypath = "./content/templates/county_data.csv"

def readable(x):
    try:
        x = "{:,}".format(int(x))
    except Exception:
        x = None
    return x
    
def percent(x):
    if x <= 1:
        return "{:.2%}".format(x)
    else:
        return "{:.2f}".format(x) + "%"

def daily(name):
    date = [] 
    new_cases = []
    new_deaths = []
    name = name.split(", ")
    county_name = name[0]
    state_name = mapping[name[1]]
    df = pd.read_csv(dailypath)
    df = df.groupby("county")
    df = df.get_group(county_name).groupby("state")
    county_data = df.get_group(state_name)
    for i in range(len(county_data) - 14, len(county_data)):
        date.append(county_data.iloc[i]['date'][5:])
        new_cases.append(county_data.iloc[i]['cases'] - county_data.iloc[i - 1]['cases'])
        new_deaths.append(county_data.iloc[i]['deaths'] - county_data.iloc[i - 1]['deaths'])
    data = {
        "date": list(county_data["date"]), 
        "cases": list(county_data["cases"]), 
        "deaths": list(county_data["deaths"]), 
        "d14_date": date, 
        "new_cases": new_cases, 
        "new_deaths": new_deaths
    }
    draw_d14("new_cases", data)
    draw_d14("new_deaths", data)
    draw_general(data)
    return {
        "cases": readable(county_data.iloc[-1]["cases"]), 
        "deaths": readable(county_data.iloc[-1]["deaths"]), 
        "death_rate": percent(county_data.iloc[-1]["deaths"] / county_data.iloc[-1]["cases"]), 
        "new_cases": readable(new_cases[-1]), 
    }

from .county_draw import draw_vac

vacpath = "./content/templates/county_vac.csv"

def vaccine(name):
    name = name.split(", ")
    county_name = name[0]
    df = pd.read_csv(vacpath)
    df["recip_county"] = df["recip_county"].apply(lambda s: s.replace(" County", ""))
    df = df.groupby("recip_county")
    df = df.get_group(county_name).groupby("recip_state")
    county_vac = df.get_group(name[1])
    vacs = {
        "full_vac_12p": county_vac.iloc[0]["series_complete_12plus"] - county_vac.iloc[0]["series_complete_18plus"], 
        "full_vac_18p": county_vac.iloc[0]["series_complete_18plus"] - county_vac.iloc[0]["series_complete_65plus"], 
        "full_vac_65p": county_vac.iloc[0]["series_complete_65plus"], 
        "d1_vac_12p": county_vac.iloc[0]["administered_dose1_recip_12plus"] - county_vac.iloc[0]["administered_dose1_recip_18plus"], 
        "d1_vac_18p": county_vac.iloc[0]["administered_dose1_recip_18plus"] - county_vac.iloc[0]["administered_dose1_recip_65plus"],
        "d1_vac_65p": county_vac.iloc[0]["administered_dose1_recip_65plus"], 
    }
    draw_vac(vacs, "full_vac")
    draw_vac(vacs, "d1_vac")
    return {
        "full_vac": readable(county_vac.iloc[0]["series_complete_yes"]), 
        "full_vac_rate": str(int(county_vac.iloc[0]["series_complete_pop_pct"])) + "%", 
        "d1_vac": readable(county_vac.iloc[0]["administered_dose1_recip"]), 
        "d1_vac_rate": str(int(county_vac.iloc[0]["administered_dose1_pop_pct"])) + "%", 
    }

def load(name, path):
    name = name.split(", ")
    name = name[0] + " County, "+ mapping[name[1]]
    df = pd.read_csv(path)
    if 'state' and 'county' in df.columns:
        df = df.drop(['state', 'county'], axis = 1)
    df = df.groupby('NAME')
    return df.get_group(name).iloc[0]

inspath = "./content/templates/county_ins_.csv"

def cook(data):
    re = list(data)
    re = re[1:]
    re = list(map(lambda x: int(x) if x != None else 0, re))
    return re

from .county_draw import draw_ins

def insurance(name):
    data = {
        '<19': cook(load(name, inspath[:-4] + '19' + inspath[-4:])), 
        '19-65': cook(load(name, inspath[:-4] + '19-65' + inspath[-4:])), 
        '>65': cook(load(name, inspath[:-4] + '65' + inspath[-4:]))
    }
    pop = {
        '<19': sum(data['<19']), 
        '19-65': sum(data['19-65']), 
        '65': sum(data['>65'])
    }
    pop_all = sum(list(pop.values()))
    for key in data.keys():
        data[key] = list(map(lambda x: x / pop_all, data[key]))
    draw_ins(data)

from .county_draw import draw_pop

poppath = "./content/templates/county_pop.csv"
povpath = "./content/templates/county_pov.csv"

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
    render(name, '_race')
    render(name, '_anc')
    re.update(render(name, '_age'))
    return re

def county_statics(name):
    statics = daily(deepcopy(name))
    statics.update(vaccine(deepcopy(name)))
    insurance(deepcopy(name))
    statics.update(population(deepcopy(name)))
    return statics