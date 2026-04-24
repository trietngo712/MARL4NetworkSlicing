import json
import os

# Data Setup
number_of_mecs = 5
number_of_links = 10

list_of_mecs = [{ "id": i , "type": "mec", "resources": {"cpu": 4} } for i in range(number_of_mecs)]
list_of_links = [{ "id": i , "type": "link", "resources": {"bandwidth": 10} } for i in range(number_of_links)]

# Combining lists
total_list = list_of_mecs + list_of_links

# The Saving Part
try:
    with open(os.path.join('configs', 'resource_config.json'), 'w') as f:
        json.dump(total_list, f, indent=4)
    print("Success! 'resource_config.json' has been created.")
except Exception as e:
    print(f"An error occurred: {e}")