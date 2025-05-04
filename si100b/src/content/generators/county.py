import requests
import pandas as pd
from datetime import datetime
from datetime import timedelta
from copy import deepcopy

data_url = "https://raw.fastgit.org/nytimes/covid-19-data/master/us-counties.csv"
geo_url = "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_050_00_20m.json"
vac_url = "https://data.cdc.gov/resource/8xkx-amqh.csv?$limit=4000"

datapath = "./content/templates/county_data.csv"
vacpath = "./content/templates/county_vac.csv"

def get_data():
    geo_data = requests.get(geo_url).json()
    county_data = pd.read_csv(data_url)
    county_data.to_csv(datapath, index = False)
    county_data = county_data.tail(28000)
    today = county_data.iloc[-1]['date']
    yesterday = datetime.strptime(today, "%Y-%m-%d") - timedelta(days = 7)
    yesterday = yesterday.strftime("%Y-%m-%d")
    group = county_data.groupby("date")
    county_data = deepcopy(group.get_group(today))
    last_data = deepcopy(group.get_group(yesterday))
    vac_data = pd.read_csv(vac_url).head(3282)
    vac_data.to_csv(vacpath, index = False)
    county_data["county"] = county_data["county"].apply(lambda x: x + ", ") + county_data["state"]
    county_data.dropna(subset = ["fips"], inplace = True)
    county_data["fips"] = county_data["fips"].apply(int)
    county_data = county_data.set_index("fips")
    last_data = last_data.set_index("fips")
    county_data['death_rate'] = county_data['deaths'] / county_data["cases"] * 100
    for i in county_data.index:
        if i in last_data.index:
            county_data.loc[i, "new_cases"] = (county_data.loc[i, "cases"] - last_data.loc[i, "cases"]) // 7
    county_data['new_cases'] = county_data['new_cases'].apply(lambda x: x if x >= 0 else 0)
    for i in vac_data.index:
        fip = vac_data.loc[i, "fips"]
        if fip != "UNK":
            fip = int(fip)
            if fip in county_data.index:
                county_data.loc[fip, "vac_total"] = vac_data.loc[i, "series_complete_yes"]
                county_data.loc[fip, "vac_rate"] = vac_data.loc[i, "series_complete_pop_pct"] 
    geo_dic = county_data["county"]
    return geo_data, geo_dic, county_data

def fmt(data): return "{:,}".format(data) if data != None else "None"

def init():
    geo_data, geo_dic, county_data = get_data()
    case_data = county_data["cases"].dropna().apply(int)
    death_data = county_data["deaths"].dropna().apply(int)
    new_case_data = county_data["new_cases"].dropna().apply(int)
    vac_total_data = county_data["vac_total"].dropna().apply(int)
    death_rate_data = county_data["death_rate"].dropna()
    vac_rate_data = county_data["vac_rate"].dropna()
    for state in geo_data['features']:
        if state['properties']['STATE'] in state_map.keys():
            state['properties']['html'] = \
                '<a href="/county/' + state['properties']['NAME'] + ', ' + state_map[state['properties']['STATE']][1] + '" target="_blank">Click for more information</a>'
            state['properties']['NAME'] += ", " + state_map[state['properties']['STATE']][0]
        pos = int(state["properties"]["GEO_ID"][-5:])
        state['properties']['cases'] = fmt(case_data.get(pos, None))
        state['properties']['deaths'] = fmt(death_data.get(pos, None))
        state['properties']['new_cases'] = fmt(new_case_data.get(pos, None))
        state['properties']['vac_total'] = fmt(vac_total_data.get(pos, None))
        state['properties']['death_rate'] = ("%.1f"%death_rate_data.get(pos, None)) + "%"\
            if death_rate_data.get(pos, None) != None else "None"
        if vac_rate_data.get(pos, None) != None:
            if vac_rate_data.get(pos, None) != 0:
                state['properties']['vac_rate'] = ("%.1f"%vac_rate_data.get(pos, None)) + "%"
            else:
                state['properties']['vac_rate'] = "None"
                vac_rate_data.drop(pos, inplace=True)
        else:
            state['properties']['vac_rate'] = "None"
    datadir = {
        "cases": case_data, 
        "deaths": death_data, 
        "new_cases": new_case_data, 
        "death_rate": death_rate_data, 
        "vac_rate": vac_rate_data, 
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
    "PuBuGn": branca.colormap.linear.PuBuGn_09
}
slogan = {
    "cases": "Confirmed Cases", 
    "deaths": "Confirmed Deaths", 
    "new_cases": "New Cases", 
    "death_rate": "Death Rate", 
    "vac_rate": "Fully-Vaccination Rate", 
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
    amap = folium.Map(
        location = [35.3, -97.6], 
        zoom_start = 4, 
        tiles = "cartodbpositron", 
        min_zoom = 3, 
        max_zoom = 10
    )
    global geo_data
    colorscale = colormap[colorset].scale(0, maxn)
    folium.GeoJson(
        name = name, 
        data = geo_data, 
        popup = folium.GeoJsonPopup(
            fields = ["NAME", popup, "html"], 
            aliases = ["County", slogan[popup], "Link"]
        ),
        style_function = lambda feature: {
            "fillOpacity": 0.7,
            "weight": 0, 
            "fillColor": "#545454" \
                if data.get(int(feature["properties"]["GEO_ID"][-5:]), None) is None \
                else colorscale(data.get(int(feature["properties"]["GEO_ID"][-5:]), None))
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

def raw_name(s):
    s = list(s.split(","))
    return s[0]

state_map = {
    "01": ["Alabama","AL"], 
    "02": ["Alaska","AK"], 
    "04": ["Arizona","AZ"], 
    "05": ["Arkansas","AR"], 
    "06": ["California","CA"], 
    "08": ["Colorado","CO"], 
    "09": ["Connecticut","CT"], 
    "10": ["Delaware","DE"], 
    "11": ["District of Columbia","DC"], 
    "12": ["Florida","FL"], 
    "13": ["Georgia","GA"], 
    "15": ["Hawaii","HI"], 
    "16": ["Idaho","ID"], 
    "17": ["Illinois","IL"], 
    "18": ["Indiana","IN"], 
    "19": ["Iowa","IA"], 
    "20": ["Kansas","KS"], 
    "21": ["Kentucky","KY"], 
    "22": ["Louisiana","LA"], 
    "23": ["Maine","ME"], 
    "24": ["Maryland","MD"], 
    "25": ["Massachusetts","MA"], 
    "26": ["Michigan","MI"], 
    "27": ["Minnesota","MN"], 
    "28": ["Mississippi","MS"], 
    "29": ["Missouri","MO"], 
    "30": ["Montana","MT"], 
    "31": ["Nebraska","NE"], 
    "32": ["Nevada","NV"], 
    "33": ["New Hampshire","NH"],
    "34": ["New Jersey","NJ"], 
    "35": ["New Mexico","NM"], 
    "36": ["New York","NY"], 
    "37": ["North Carolina","NC"], 
    "38": ["North Dakota","ND"], 
    "39": ["Ohio","OH"], 
    "40": ["Oklahoma","OK"], 
    "41": ["Oregon","OR"], 
    "42": ["Pennsylvania","PA"], 
    "44": ["Rhode Island","RI"], 
    "45": ["South Carolina","SC"], 
    "46": ["South Dakota","SD"], 
    "47": ["Tennessee","TN"], 
    "48": ["Texas","TX"], 
    "49": ["Utah","UT"], 
    "50": ["Vermont","VT"], 
    "51": ["Virginia","VA"], 
    "53": ["Washington","WA"], 
    "54": ["West Virginia","WV"], 
    "55": ["Wisconsin","WI"], 
    "56": ["Wyoming","WY"], 
    "72": ["Puerto Rico", "PR"]
}

def county():
    geo_data, geo_dic, datadir = init()
    render(
        geo_data,
        ("Confirmed Cases in the United States", datadir["cases"], 
            "infect map county case", "YlOrRd", "cases", 10000), 
        ("Confirmed Deaths in the United States", datadir["deaths"], 
            "infect map county death", "YlGnBu", "deaths", 300), 
        ("New Cases in the United States (7d-Average)", datadir["new_cases"], 
            "infect map county new case", "YlOrBr", "new_cases", 75), 
        ("Death Rate in the United States(%)", datadir["death_rate"], 
            "infect map county death rate", "PuRd", "death_rate", 2.5), 
        ("Fully-Vaccination Rate in the United States(%)", datadir["vac_rate"], 
            "infect map county vac rate", "PuBuGn", "vac_rate", 75), 
        ("Fully-Vaccined Population in the United States", datadir["vac_total"], 
            "infect map county vac total", "BuGn", "vac_total", 18000)
    )
    re = {"countynames": ""}
    for state in geo_data["features"]:
        name = state['properties']['NAME'].split(",")
        name = name[0]
        if state['properties']['STATE'] in state_map.keys():
            name += ", " + state_map[state['properties']['STATE']][1]
            re["countynames"] += ('"' + name + '", ')
    cases = datadir["cases"].sort_values(ascending = False)
    deaths = datadir["deaths"].sort_values(ascending = False)
    new_cases = datadir["new_cases"].sort_values(ascending = False)
    for i in range(10):
        re[f"casesbycountyvalue{str(i)}"] = "{:,}".format(cases.iloc[i])
        re[f"casesbycountyname{str(i)}"] = raw_name(geo_dic.loc[cases.index[i]])
        re[f"deathsbycountyvalue{str(i)}"] = "{:,}".format(deaths.iloc[i])
        re[f"deathsbycountyname{str(i)}"] = raw_name(geo_dic.loc[deaths.index[i]])
        re[f"newcasesbycountyvalue{str(i)}"] = "{:,}".format(new_cases.iloc[i])
        re[f"newcasesbycountyname{str(i)}"] = raw_name(geo_dic.loc[new_cases.index[i]])
    return re