from tabulate import tabulate

def table_creation(headers, data, file):
    table = {}
    for i, h in enumerate(headers):
        table.update({h: data[i]})
    with open('./tables/' + file, 'w') as file:
        file.write(tabulate(table, headers='keys', tablefmt='fancy_grid'))
    return True

def normalize_data(data):
    return (data - data.mean()) / \
           data.std()

