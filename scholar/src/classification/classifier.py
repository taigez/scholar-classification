from utils import scrape_google, split_sen, get_text, multiclass

def scholar_search(name):
    final_dict = {}
    awd = []
    edu = []
    interest = []
    position = []

    for link in scrape_google(name):
        texts = split_sen(get_text(link))
        if texts is not None:
            for t in texts:
                pred = multiclass(t)
                if pred == 0:
                    awd.append(t)
                elif pred == 1:
                    edu.append(t)
                elif pred == 2:
                    interest.append(t)
                elif pred == 3:
                    position.append(t)
        else:
            continue
    final_dict["awd"] = awd
    final_dict["edu"] = edu
    final_dict["int"] = interest
    final_dict["pos"] = position
    return final_dict

def url_search(urls):
    final_dict = {}
    awd = []
    edu = []
    interest = []
    position = []

    for link in urls:
        texts = split_sen(get_text(link))
        if texts is not None:
            for t in texts:
                pred = multiclass(t)
                if pred == 0:
                    awd.append(t)
                elif pred == 1:
                    edu.append(t)
                elif pred == 2:
                    interest.append(t)
                elif pred == 3:
                    position.append(t)
        else:
            continue

    final_dict["awd"] = awd
    final_dict["edu"] = edu
    final_dict["int"] = interest
    final_dict["pos"] = position

    return final_dict