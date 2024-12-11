def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "bic":
        from models.bic import BiC
        return BiC(args)
    elif name == "podnet":
        from models.podnet import PODNet
        return PODNet(args)
    elif name == "lwf":
        from models.lwf import LwF
        return LwF(args)   
    elif name == "wa":
        from models.wa import WA
        return WA(args)
    elif name == "der":
        from models.der import DER
        return DER(args)
    elif name == "replay":
        from models.replay import Replay
        return Replay(args)
    else:
        assert 0
