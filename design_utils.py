import pandas as pd


def stringify_attributes(attributes={}):
    return ' '.join([f'{attr}="{attributes[attr]}"' for attr in attributes])

def add_content(items, attributes={}, extra=''):
    return f"{extra} <ul>"+' '.join([f"<li {stringify_attributes(attributes)}>{item}</li>" for item in items]) + "</ul>" 

add_listitem = lambda items, extra='': add_content(items, {'style':'padding-right:10px; padding-bottom:10px; font-family:"Poppins";'}, extra=extra)

wrap_title = lambda content:f"* <h3><b>{content}</b><h3>"

def add_resource(url, title, content, img_src, img_width=100, img_height=200):
    return f"""
                    <div style="display:grid; grid-template-columns: repeat(9, 1fr); margin-bottom:20px" >        
                    <div style=" grid-column: 1/ 4; padding: 5px;">
                        <a  href="{url}"> {title}</a> 
                        <p style="overflow-wrap: break-word; word-wrap: break-word; hyphens: auto;">
                            {content}
                        </p>                        
                    </div>

                    <img src="{img_src}" 
                        width={img_width} height={img_height} style="grid-column: 4/ -1;"/>
                </div>
    """

def wrap_with_html_tag(tag, content, attributes={}):
    return f"""<{tag} {stringify_attributes(attributes)}>
             {content}
            </{tag}>
           """


def create_table(items:list[list[str]]):
    res = f''
    for item in items:
        res += wrap_with_html_tag("tr", " ".join([ wrap_with_html_tag("td",p) for p in item]))
    return wrap_with_html_tag("table", res)    

def ms_to_fps(x):
    return f"{int(1/(x*10**(-3)))} fps"

def list_to_table(ls):
    table = pd.DataFrame(ls[1:], columns=ls[0])
    return table

CONFIG = {
    "n_tables": 0
}
def make_table_title(text:str, url=""):
    CONFIG["n_tables"] +=1

    return wrap_with_html_tag('h6', f"{CONFIG['n_tables']}. "+text, {"style":'text-align: center; transform:translateY(-30px);',
                                            "href":url})
