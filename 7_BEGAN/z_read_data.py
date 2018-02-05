import csv,sys
import requests
i = 0
with open('gif_data/short_data.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:

        with open('gif_data_download/'+str(i)+'_'+str(row[1])+'.gif', 'wb') as f:
            f.write(requests.get(row[0]).content)
        
        i = i + 1
        if i == 500:
            print("stop")
            sys.exit()
