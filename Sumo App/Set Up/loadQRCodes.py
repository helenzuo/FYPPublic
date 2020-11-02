import json
import pandas as pd

with open('../stations.txt') as dataFile:
    df = pd.read_excel(r'D:\Google Drive\Uni\FYP\ClientApp\QR.xlsx', sheet_name='Sheet1')
    dataLoaded = json.load(dataFile)
    for ind in range(len(dataLoaded)):
        dataLoaded[-1-ind]['QR'] = int(df["Codes"][ind])
    print(dataLoaded)

with open('../stations.txt', 'w') as dataFile:
    json.dump(dataLoaded, dataFile, ensure_ascii=False)