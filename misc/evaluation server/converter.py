import os
import json

base_path = r"" # The directory containing your .txt file.
destination_path = r"" # the folder where you wish to store the resulting .json file.
saved_name = "" # You can customize the name of the .json file as desired.



def detect_and_split(line):
    if ',' in line and ' ' not in line:
        return line.split(',')
    else:
        return line.split()

saved_name = os.path.join(destination_path,saved_name+".json")
pred_data = {}


for filename in os.listdir(base_path):
    if filename.endswith('.txt'):

        base_filename = filename[:-4]
        if "time" in base_filename:
            base_filename = "time"

        content_list = []
        with open(os.path.join(base_path, filename), 'r') as file:
            content = file.readlines()
            if base_filename != "time":
    
                for line in content:        
                    parts = detect_and_split(line)
                    content_list.append(list(map(lambda x: int(float(x)), parts)))
            else:
                content_list = [float(i.split()[0]) for i in content]


            if base_filename not in pred_data:
                if base_filename !="time":
                    pred_data[base_filename] = {"pred": []}
                else:
                    pred_data[base_filename] = []
                    
            if base_filename != 'time':
                pred_data[base_filename]["pred"] = content_list
            else:
                pred_data[base_filename].append(content_list)
if "time" not in pred_data.keys():
    pred_data["time"] = [[-211000]]
            

with open(saved_name, 'w', encoding='utf-8') as json_file:
    json.dump(pred_data, json_file, indent=4)

print(saved_name," saved!")