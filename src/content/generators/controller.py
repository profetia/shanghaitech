import time
import json
import pandas as pd
from .d14_statics import d14_statics
from .d30_statics import d30_statics
from .d90_statics import d90_statics
from .general_statics import general_statics
from .predic_statics import predic_statics
from .county import county
from .state import state

url = "https://raw.fastgit.org/nytimes/covid-19-data/master/us.csv"

def statics():
    data = pd.read_csv(url)
    general_statics(data)
    re = {
        "totalcases": "{:,}".format(data.iloc[-1]["cases"]), 
        "totaldeaths": "{:,}".format(data.iloc[-1]["deaths"]), 
        "totalnewcases": "{:,}".format(data.iloc[-1]["cases"] - data.iloc[-2]["cases"])
    }
    re.update(d14_statics(data))
    re.update(d30_statics(data))
    re.update(d90_statics(data))
    re.update(predic_statics(data))
    return re

filename = "./content/templates/statics.json"

def controller():
    dic = {
        "casesbycounty": "/images/casesbycounty", 
        "deathsbycounty": "/images/deathsbycounty", 
        "deathratebycounty": "/images/deathratebycounty", 
        "newcasesbycounty": "/images/newcasesbycounty", 
        "vactotalbycounty": "/images/vactotalbycounty", 
        "vacratebycounty": "/images/vacratebycounty", 
        "casesbystate": "/images/casesbystate", 
        "deathsbystate": "/images/deathsbystate", 
        "deathratebystate": "/images/deathratebystate", 
        "newcasesbystate": "/images/newcasesbystate", 
        "vactotalbystate": "/images/vactotalbystate", 
    }
    now = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    dic["updtime"] = now
    dic.update(county()) 
    dic.update(state()) 
    dic.update(statics())
    with open(filename, "w") as f:
        json.dump(dic, f)