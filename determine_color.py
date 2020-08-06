import os
from DetermineColor import DetermineColor
import json

PATH = "/home/ilyas/local_disk/Py/ELAN/clothes_classification_by_color_opencv/images/"
color_determiner = DetermineColor()
n = os.listdir(PATH)
n = [i for i in n if not i.endswith("json")]
for i in range(len(n)):
    name = PATH + n[i]
    colors_dict = color_determiner.run(name)
    keys_dict = list(colors_dict.keys())
    result = {}
    for j in range(len(keys_dict)):
        color = keys_dict[j]
        value = colors_dict[color]['value']
        result[color] = value
    with open(name + '.json', 'w') as f:
        f.write(json.dumps(result))
