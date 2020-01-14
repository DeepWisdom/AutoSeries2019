from .feat_gen import *



class FeatPipeline:
    def __init__(self):
        self.order1s = [
            KeysCross,
            TimeDate,
            Delta,
            #KeysTimeCross,
            #GroupMean,
            LagFeat,
            #TimeSinceLast
            #WindowMinus,
            #WindowEncode,
        ]
