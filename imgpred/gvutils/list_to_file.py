import json

def list_to_file(filename, mlist, mode="w"):
    with open(filename, mode) as file_in:
        # for item in mlist:
        #     file_in.write("%s\n" % item)
        file_in.write(json.dumps(mlist))
