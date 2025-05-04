import time
import json
import pandas as pd
from datetime import datetime
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_exempt
from apscheduler.schedulers.background import BackgroundScheduler
from .generators.controller import controller
from .generators.county_statics import county_statics
from .generators.county_statics import full_state
from .generators.state_statics import state_statics
# Create your views here.

jsonname = "./content/templates/statics.json"

def index(request):
    dic = {}
    with open(jsonname) as f:
        dic = json.load(f)
    return render(request, "index.html", dic)

csvname = './content/templates/county_data.csv'

def find_county(request):
    name = request.path[8:]
    name = name.replace("%20", " ")
    ref = pd.read_csv(csvname).tail(4000)
    arr = name.split(", ")
    county_name = arr[0]
    state_name = arr[1]
    if state_name == "PR":
        return render(request, "error.html", {"slogan": "SERVICE UNAVAILABLE"})
    else:
        state_name = full_state(state_name)
        ref = ref.groupby("state").get_group(state_name)
        if county_name not in list(ref['county']):
            return render(request, "error.html", {"slogan": "NO DATA"})
        else:
            dic = {"full_name": county_name + ", " + state_name}
            dic.update(county_statics(name))
            return render(request, "county.html", dic)

def find_state(request):
    name = request.path[7:]
    name = name.replace("%20", " ")
    ref = pd.read_csv(csvname).tail(4000)
    if name == "Puerto Rico":
        return render(request, "error.html", {"slogan": "SERVICE UNAVAILABLE"})
    else:
        if name not in list(ref['state']):
            return render(request, "error.html", {"slogan": "NO DATA"})
        else:
            dic = {"full_name": name}
            dic.update(state_statics(name))
            return render(request, "state.html", dic)

logsname = "./content/logs.txt"

def get_logs(request):
    re = ""
    with open(logsname) as f:
        lines = f.readlines()
        for line in lines:
            re += "<p>" + line + "</p>"
    return HttpResponse(re)

image_map = {
    "casesbycounty": "infect map county case.html", 
    "deathsbycounty": "infect map county death.html", 
    "deathratebycounty": "infect map county death rate.html", 
    "newcasesbycounty": "infect map county new case.html", 
    "vactotalbycounty": "infect map county vac total.html", 
    "vacratebycounty": "infect map county vac rate.html", 
    "casesbystate": "infect map state case.html", 
    "deathsbystate": "infect map state death.html", 
    "deathratebystate": "infect map state death rate.html", 
    "newcasesbystate": "infect map state new case.html", 
    "vactotalbystate": "infect map state vac total.html", 
}

@xframe_options_exempt
def images(request):
    name = request.path[8:]
    return render(request, image_map[name])

filename = "./content/logs.txt"

def update():
    try:
        controller()
        with open(filename, "a") as f:
            now = time.asctime(time.localtime())
            f.write(now + " update successful\n\n")
    except Exception as e:
        with open(filename, "a") as f:
            now = time.asctime(time.localtime())
            f.write(now + " update failed " + str(e) + "\n\n")

scheduler = BackgroundScheduler()
scheduler.add_job(update, 'interval', id="auto-update", 
                    replace_existing = True, next_run_time = datetime.now(), 
                    hours = 4)
try:
    scheduler.start()
    with open(filename, "a") as f:
        now = time.asctime(time.localtime())
        f.write(now + " auto-update start successful\n\n")
except Exception as e:
    with open(filename, "a") as f:
        now = time.asctime(time.localtime())
        f.write(now + " auto-update start failed " + str(e) + "\n\n")
