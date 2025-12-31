import polaris as po
import json
import pandas as pd

def main(json=True, txt=True):
    '''
    Function extracts data from polaris dataset object and writes
    each entry as a dict in json and/or txt format.
    '''

    dataset = po.load_dataset("asap-discovery/antiviral-potency-2025-unblinded")

    # list of dict objects with each datapoint as a dict
    data_dicts = [dataset[i] for i in range(len(dataset))] #this takes half an age

    if json:
        #writes json file as list of dicts 
        with open("./data/dataset.json", "w", encoding = "utf-8") as f:
            json.dump(data_dicts, f, indent = 2, ensure_ascii=False)

    if txt:
        #writes txt file with each line as a dict
        with open("./data/dataset.txt", "w", encoding="utf-8") as f:
            for item in data_dicts:
                f.write(json.dumps(item, ensure_ascii=False)+ "\n")

dataset = [] # list of dict objects -- mirror of `data_dicts`

with open("./data/dataset.txt") as f:
    for line in f:
        data = json.loads(line.strip())
        dataset.append(data)

dataset_df = pd.DataFrame(dataset) # pandas dataframe of the dataset

# only 
if __name__ == '__main__':
    main()