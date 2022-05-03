import requests
from bs4 import BeautifulSoup

generate_url = lambda s:f"https://www.google.com/search?q={s}&sxsrf=ALiCzsblJUPfLDI3O5QKcEr89i8up-IYEQ:1651494589361&source=lnms&tbm=isch&sa=X&ved=2ahUKEwirr-Pi6MD3AhWJEKYKHSsaBZsQ_AUoAXoECAIQAw&biw=730&bih=767&dpr=1.25"

def collect_img_from_google_search(search):
    url = generate_url(search)
    srcs  = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    elements = soup.findAll("img")
    for element in elements:
        src = element.get('src')
        if src is None: continue
        if "https" in src:
            srcs.append(src)
    return srcs