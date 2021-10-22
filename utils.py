import json, pickle, re

class Namespace:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

def get_config(config_file): 
  with open(config_file) as f:
        config = json.load(f)
  n = Namespace()
  n.__dict__.update(config)
  return n

def load(datadir):
    with open(datadir, encoding='utf-8') as f:
        data = f.read().splitlines()
    return data

def write(data, savedir, mode='w'):
    f = open(savedir, mode)
    for text in data:
        f.write(text+'\n')
    f.close()


def load_pickle(datadir):
  file = open(datadir, 'rb')
  data = pickle.load(file)
  return data

def write_pickle(data, savedir):
  file = open(savedir, 'wb')
  pickle.dump(data, file)
  file.close()

def load_json(datadir):
    with open(datadir, 'rb') as file:
        return json.load(file)

def write_json(data, savedir):
    with open(savedir, 'w') as file:
        json.dump(data, file, indent=4)


def print_dict(d, limit=20):
    cnt = 0
    for k, v in d.items():
        print(f'{k} : {v}')
        print('-' * 30)
        if cnt > limit:
            break
        cnt += 1 

def get_dict_index(example):
    try:
        page_index = re.search(r'(\w)', example).group(0).lower()
        word = re.search(r'(\w+)', example).group(0).lower()
        return page_index, word
    except:
        tokens = example.split(' ')
        return tokens[0][0].lower(), tokens[0].lower() 