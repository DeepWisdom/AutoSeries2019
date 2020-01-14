from .feat_pipeline import FeatPipeline
from log_utils import timeclass

class FeatEngine:
    def __init__(self, feat_pipeline: FeatPipeline):
        self.feat_pipeline = feat_pipeline

    #@timeclass(cls='FeatEngine')
    def fit_oder1s(self, table):
        self.feats_order1s = []
        for feat_cls in self.feat_pipeline.order1s:
            feat = feat_cls()
            feat.fit(table)
            self.feats_order1s.append(feat)


    #@timeclass(cls='FeatEngine')
    def transform_oder1s(self, table):
        for feat in self.feats_order1s:
            feat.transform(table)

    #@timeclass(cls='FeatEngine')
    def fit_transform_oder1s(self, table):
        self.feats_order1s = []
        for feat_cls in self.feat_pipeline.order1s:
            feat = feat_cls()
            feat.fit_transform(table)
            self.feats_order1s.append(feat)