import sys

sys.path.insert(0, ".")

import quant

df = quant.data.amfi(28, "151739")
print(df)
