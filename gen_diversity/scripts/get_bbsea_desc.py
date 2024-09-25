import os
import yaml

group = "table"
root_path = "/localdata/bxu/isaac_lab/bbsea/your_path_to_output"

group_path = os.path.join(root_path, group)
descs = []
for task in os.listdir(group_path):
    desc = task.split("_")[1]
    descs.append(desc)
print(descs)

yaml.safe_dump(descs, open(f"{group}_descs.yaml", "w"))