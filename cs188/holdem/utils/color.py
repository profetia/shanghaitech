# colors = ['#72A98F', '#F1C40F', '#E67E22', '#C0392B', '#3498DB']
colors = ['#3498DB', '#C0392B', '#E67E22', '#F1C40F', '#72A98F']

dict_color = {
}

used_count = 0

def use_color(name:str):
    global used_count
    if name not in dict_color:
        dict_color[name] = colors[used_count]
        used_count = (used_count + 1) % len(colors)
    return dict_color[name]