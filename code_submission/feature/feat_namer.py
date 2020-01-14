import CONSTANT

class FeatNamer:
    @staticmethod
    def gen_feat_name(cls_name, feat_name, param, feat_type):
        prefix = CONSTANT.type2prefix[feat_type]
        if param == None:
            return f"{prefix}{cls_name}:{feat_name}"
        else:
            return f"{prefix}{cls_name}:{feat_name}:{param}"

    @staticmethod
    def gen_merge_name(table_name,feat_name,feat_type):
        prefix = CONSTANT.type2prefix[feat_type]
        return f"{prefix}{table_name}.({feat_name})"

    @staticmethod
    def gen_merge_feat_name(cls_name, feat_name, param, feat_type, table_name):
        feat_name = FeatNamer.gen_feat_name(cls_name, feat_name, param, feat_type)
        return FeatNamer.gen_merge_name(table_name, feat_name, feat_type)

