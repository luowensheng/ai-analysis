from subprocess import check_output
import re

def install_packages():
    pattern = "ModuleNotFoundError: No module named \'\w+\'"
    # output = check_output("venv/Scripts/activate")
    try:
        output = check_output("python main.py")
    except Exception as e:
        # print(output)
        print(e.with_traceback(None))
        res = re.findall(pattern, e.with_traceback(None))
        # print(res)
    # print(output)


install_packages()