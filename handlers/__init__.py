from handlers import adain_handler_balanced
from handlers.adain_handler_balanced import StyleAlignedArgs


FALSE_SA_ARGS = StyleAlignedArgs(share_group_norm=False,
                                    share_layer_norm=False,
                                    share_attention=False,
                                    adain_queries=False,
                                    adain_keys=False,
                                    adain_values=False)

def get_handler():
    return adain_handler_balanced.Handler