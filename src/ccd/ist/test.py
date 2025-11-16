from .transfer import StyleTransfer
from pathlib import Path
import os
import json

lang = "java"
ist = StyleTransfer(lang)
# * -1.1 -3.1 0.5 3.4 4.4 7.2 8.1 9.1 10.7
style = ("-1.1","-3.1")
code = Path(f"/home/nfs/u2023-zlb/CauDefend/src/ccd/ist/test/test.java").read_text()
pcode, succ = ist.transfer(code=code, styles=[style])
print(f"succ = {succ}")
print(pcode)

print(
    f"{ist.get_style(code=code, styles=[style])[style]} -> {ist.get_style(code=pcode, styles=[style])[style]}"
)