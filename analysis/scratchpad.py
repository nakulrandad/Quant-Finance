import sys

sys.path.insert(0, ".")

import quant as qt

df = qt.api.mf_list(filter=["Growth", "Direct", "SBI"])
print(df)
