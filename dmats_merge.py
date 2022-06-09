import sys, json

file_name1 = sys.argv[1]
file_name2 = sys.argv[2]

# read in both files

file1 = open(file_name1, 'r')
file2 = open(file_name2, 'r')

dict_out = json.load(file1)
dict_in = json.load(file2)

file1.close()
file2.close()

for n in dict_in.keys():
    for p in dict_in[n].keys():
        for q in dict_in[n][p].keys():
            for metric in dict_in[n][p][q].keys():
                if n in dict_out.keys():
                    if p in dict_out[n].keys():
                        if q in dict_out[n][p].keys():
                            if not metric in dict_out[n][p][q].keys():
                                dmats = dict_in[n][p][q][metric]
                                dict_out[n][p][q].update({metric: dmats})
                        else:
                            dmats = dict_in[n][p][q][metric]
                            dict_out[n][p].update({q: {metric: dmats}})
                    else:
                        dmats = dict_in[n][p][q][metric]
                        dict_out[n].update({p: {q: {metric: dmats}}})
                else:
                    dmats = dict_in[n][p][q][metric]
                    dict_out.update({n: {p: {q: {metric: dmats}}}})

# write output to FIRST file

json_file = open(file_name1, "w")
json.dump(dict_out, json_file)
json_file.close()
