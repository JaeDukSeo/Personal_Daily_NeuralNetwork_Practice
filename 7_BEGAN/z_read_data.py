import csv,sys

with open('gif_data/short_data.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        print(row[0])
        print(row[1])
        

        sys.exit(0)