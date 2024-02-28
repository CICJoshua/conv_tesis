""" 

    dataFrame.py ---------> Clase para devolver un dataFrame límpio...

"""

import pandas as df

class DataFrame:

    def __init__(self) -> None:
        pass
    def __del__(self) -> None:
        print("Obj deleted")

    def getDataframe(self, __file):
        print(__file)
        pd = df.read_json(__file)
        return pd.T

    