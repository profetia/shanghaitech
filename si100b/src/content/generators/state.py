import requests
import pandas as pd
from datetime import datetime
from datetime import timedelta
from copy import deepcopy

data_url = "https://raw.fastgit.org/nytimes/covid-19-data/master/us-states.csv"
geo_url = "https://raw.fastgit.org/HuStanding/show_geo/master/us-states.json"
vac_url = "https://raw.fastgit.org/govex/COVID-19/master/data_tables/vaccine_data/us_data/time_series/people_vaccinated_us_timeline.csv"

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

datapath = "./content/templates/state_data.csv"
vacpath = "./content/templates/state_vac.csv"

def get_data():
    geo_data = requests.get(geo_url).json()
    state_data = pd.read_csv(data_url)
    state_data.to_csv(datapath, index = False)
    state_data = state_data.tail(480)
    today = state_data.iloc[-1]['date']
    yesterday = datetime.strptime(today, "%Y-%m-%d") - timedelta(days = 7)
    yesterday = yesterday.strftime("%Y-%m-%d")
    group = state_data.groupby("date")
    state_data = deepcopy(group.get_group(today))
    last_data = deepcopy(group.get_group(yesterday))
    state_id = deepcopy(mapping)
    vac_data = pd.read_csv(vac_url)
    state_id = dict(zip(state_id.values(), state_id.keys()))
    vac_data = vac_data.sort_values(by = ["Date"]).tail(61)
    vac_data.to_csv(vacpath, index = False)
    state_data['id'] = state_data['state'].apply(
        lambda x: state_id[x] if x in state_id.keys() else pd.NA)
    last_data['id'] = last_data['state'].apply(
        lambda x: state_id[x] if x in state_id.keys() else pd.NA)   
    state_data.dropna(subset = ["id"], inplace = True)
    state_data.set_index("id", inplace = True)
    last_data.set_index("id", inplace = True)
    state_data["death_rate"] = state_data["deaths"] / state_data["cases"] * 100
    for i in state_data.index:
        if i in last_data.index:
            state_data.loc[i, "new_cases"] = (state_data.loc[i, "cases"] - last_data.loc[i, "cases"]) // 7
    state_data['new_cases'] = state_data['new_cases'].apply(lambda x: x if x >= 0 else 0)
    for i in vac_data.index:
        name = vac_data.loc[i, "Province_State"]
        if name in state_id.keys():
            fip = state_id[name]
            state_data.loc[fip, "vac_total"] = vac_data.loc[i, "People_Fully_Vaccinated"]
    geo_dic = state_data["state"]
    return geo_data, geo_dic, state_data

def fmt(data): return "{:,}".format(data) if data != None else "None"

def init():
    geo_data, geo_dic, state_data = get_data()
    case_data = state_data["cases"].dropna().apply(int)
    death_data = state_data["deaths"].dropna().apply(int)
    new_case_data = state_data["new_cases"].dropna().apply(int)
    vac_total_data = state_data["vac_total"].dropna().apply(int)
    death_rate_data = state_data["death_rate"]
    for state in geo_data['features']:
        pos = state["id"]
        state['properties']['html'] = \
            '<a href="/state/' + state['properties']['name'] + '" target="_blank">Click for more information</a>'
        state['properties']['cases'] = fmt(case_data.get(pos, None))
        state['properties']['deaths'] = fmt(death_data.get(pos, None))
        state['properties']['new_cases'] = fmt(new_case_data.get(pos, None))
        state['properties']['vac_total'] = fmt(vac_total_data.get(pos, None))
        state['properties']['death_rate'] = ("%.1f"%death_rate_data.get(pos, None)) + "%"\
            if death_rate_data.get(pos, None) != None else "None"
    case_data = state_data["cases"].dropna().apply(lambda x: round(x / 1000, 1))
    vac_total_data = state_data["vac_total"].dropna().apply(lambda x: round(x / 1000, 1))
    datadir = {
        "cases": case_data, 
        "deaths": death_data, 
        "new_cases": new_case_data, 
        "death_rate": death_rate_data, 
        "vac_total": vac_total_data 
    }
    return geo_data, geo_dic, datadir

import folium
import branca
from threading import Thread

save_path = "./content/templates"
colormap = {
    "YlOrRd": branca.colormap.linear.YlOrRd_09, 
    "YlGnBu": branca.colormap.linear.YlGnBu_09, 
    "YlOrBr": branca.colormap.linear.YlOrBr_09, 
    "PuRd": branca.colormap.linear.PuRd_09, 
    "BuGn": branca.colormap.linear.BuGn_09, 
}
slogan = {
    "cases": "Confirmed Cases", 
    "deaths": "Confirmed Deaths", 
    "new_cases": "New Cases", 
    "death_rate": "Death Rate", 
    "vac_total": "Fully-Vaccinated Population" 
}

class draw_thread(Thread):
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)

def draw(name, data, file, colorset, popup, maxn):
    global geo_data
    amap = folium.Map(
        location = [35.3, -97.6], 
        zoom_start = 4, 
        tiles = "cartodbpositron", 
        min_zoom = 3, 
        max_zoom = 10
    )
    colorscale = colormap[colorset].scale(0, maxn)
    folium.GeoJson(
        name = name, 
        data = geo_data, 
        popup = folium.GeoJsonPopup(
            fields = ["name", popup, "html"], 
            aliases = ["State", slogan[popup], "Link"]
        ),
        style_function = lambda feature: {
            "fillOpacity": 0.7, 
            "weight": 0, 
            "fillColor": "#545454" if data.get(feature["id"], None) is None \
                        else colorscale(data.get(feature["id"], None))
        }
    ).add_to(amap)
    colorscale.caption = name
    colorscale.add_to(amap)
    filename = f"{save_path}/{file}.html"
    amap.save(filename)

def render(geo, *args):
    global geo_data
    geo_data = geo
    for param in args:
        now = draw_thread(draw, param)
        now.start()

def state():
    geo_data, geo_dic, datadir = init()
    render(
        geo_data, 
        ("Confirmed Cases in the United States (x1000)", datadir["cases"], 
            "infect map state case", "YlOrRd", "cases", 1000), 
        ("Confirmed Deaths in the United States", datadir["deaths"], 
            "infect map state death", "YlGnBu", "deaths", 18000), 
        ("New Cases in the United States (7d-Average)", datadir["new_cases"], 
            "infect map state new case", "YlOrBr", "new_cases",2500), 
        ("Death Rate in the United States(%)", datadir["death_rate"], 
            "infect map state death rate", "PuRd", "death_rate", 2.5), 
        ("Fully-Vaccined Population in the United States (x1000)", datadir["vac_total"], 
            "infect map state vac total", "BuGn", "vac_total", 3000)
    )
    re = {"statenames": ""}
    for i in range(len(geo_dic)):
        re["statenames"] += ('"' + geo_dic.iloc[i] + '", ')
    cases = datadir["cases"].sort_values(ascending = False)
    deaths = datadir["deaths"].sort_values(ascending = False)
    new_cases = datadir["new_cases"].sort_values(ascending = False)
    for i in range(10):
        re[f"casesbystatevalue{str(i)}"] = "{:,}".format(int(cases.iloc[i] * 1000))
        re[f"casesbystatename{str(i)}"] = geo_dic.loc[cases.index[i]]
        re[f"deathsbystatevalue{str(i)}"] = "{:,}".format(deaths.iloc[i])
        re[f"deathsbystatename{str(i)}"] = geo_dic.loc[deaths.index[i]]
        re[f"newcasesbystatevalue{str(i)}"] = "{:,}".format(new_cases.iloc[i])
        re[f"newcasesbystatename{str(i)}"] = geo_dic.loc[new_cases.index[i]]
    return re