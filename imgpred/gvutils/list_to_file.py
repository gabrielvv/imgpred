import json
import os

# https://stackoverflow.com/questions/12994442/appending-data-to-a-json-file-in-python
def list_to_file(fname, mlist=[], mode="+"):
    feeds = []
    if os.path.isfile(fname):
        with open(fname) as feedsjson:
            feeds = json.load(feedsjson)
    with open(fname, "w") as file_in:
        for item in mlist:
            feeds.append(item)
        file_in.write(json.dumps(feeds))
