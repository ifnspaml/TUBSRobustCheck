from collections import namedtuple
import json
import importlib
import torch
import sys
import os


# Definitions
# Model and all meta information
Model = namedtuple('Model', [

    'name',                 # The identifier of this model architecture variant, e.g. 'erfnet', ...
                            # We use them to uniquely name a model of a specific architecture.

    'id',                   # An integer ID that is associated with each model.

    'ModuleName',           # Name of file/module that includes 'cooking recipe' to create new instance of the model
                            # architecture as a pytorch nn.Module.

    'instantClass',         # The name of the class that needs to be called for instantiation.

    'modelDefLoad',         # The name of the load model definition function

    'modelStateLoad',       # The name of the file which includes parameters (weights etc.) of a previous state
                            # (e.g. after training) which is needed to load
                            # a default checkpoint.
    ])


# A list of all models
models = [
    #      name            id        ModuleName                                                                         instanceClass           modelDefLoad                  modelStateLoad
    Model('sample',        0,       'ModuleName',                                                                       'ClassName',            'LoadDefFunctionName',        'LoadStateFunctionName'),
    Model('swiftnet',      2,       'swiftnet.models.swiftnet.fw_adaptor',                                              'SwiftNet_ss',          'Load_model_swiftnet',        'Load_state_swiftnet'),
]


# Create dictionaries for a fast lookup (Please refer to the main method below for example usages!)
# Function to transform from name to modul object
name2model = {model.name: model for model in models}

# Function to transform from Id to model object
id2model = {model.id: model for model in models}


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


# Erfnet model building
def Load_model_erfnet(class_, args):
    if args.ignore_bkg:
        model = class_(num_classes=args.num_classes, encoder=args.encoder)
    else:
        model = class_(num_classes=args.num_classes+1, encoder=args.encoder)

    return model


# Erfnet state loading
def Load_state_erfnet(model, module, model_weight_path):
    print("Loading weights: " + model_weight_path, flush=True)
    print('========== state_dict() key replacement ==========')
    pretrained_dict = torch.load(model_weight_path)                                                                     # load the weights
    key_list_random = sorted([k for k, v in model.state_dict().items() if
                            "num_batches_tracked" not in k])                                                            # Create lists of keys from model.state_dict() and pretrained_dict
    key_list_pretrained = sorted([k for k, v in pretrained_dict.items()])
    new_pretrained_dict = {}                                                                                            # Map the keys from pretrained_dict to model.state_dict() keys

    for key in zip(key_list_random, key_list_pretrained):
        print(key[0], '----->', key[1])
        new_pretrained_dict[key[0]] = pretrained_dict[key[1]]
    pretrained_dict = new_pretrained_dict
    print('=================================================')

    model = load_my_state_dict(model, pretrained_dict)  # load model state

    return model


# Swiftnet model building
def Load_model_swiftnet(class_, args):
    if args.ignore_bkg:
        model = class_(args.num_classes_wo_bg, k_Logits=args.k_Logits, enc_act_fn=args.enc_act_fn,
                       spp_act_fn=args.spp_act_fn, dec_act_fn=args.dec_act_fn, DUpsampling=args.DUpsampling)
    else:
        model = class_(args.num_classes_wo_bg+1, k_Logits=args.k_Logits, enc_act_fn=args.enc_act_fn,
                       spp_act_fn=args.spp_act_fn, dec_act_fn=args.dec_act_fn, DUpsampling=args.DUpsampling)

    return model


# Swiftnet state loading
def Load_state_swiftnet(model, module, model_weight_path):
    # Load weights
    if torch.cuda.is_available():
        pretrained_dict = torch.load(model_weight_path)
    else:
        pretrained_dict = torch.load(model_weight_path, map_location='cpu')

    # Get the class, will raise AttributeError if class cannot be found
    function_ = getattr(module, 'load_state_dict_into_model')

    # Call custom function to load pretrained state dict for the specific model architecture
    function_(model, pretrained_dict)

    return model


# Deeplab model building
def Load_model_deeplab(class_, args):
    model = class_(backbone=args.backbone, output_stride=args.output_stride, num_classes=args.num_classes,
                   sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)

    return model


# Deeplab state loading
def Load_state_deeplab(model, module, model_weight_path):
    # Load weights
    if torch.cuda.is_available():
        checkpoint = torch.load(model_weight_path)
    else:
        checkpoint = torch.load(model_weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})" .format(model_weight_path, checkpoint['epoch']))

    return model


# Load model definition
def load_model_def(model_name):
    # Load the module which includes model def; will raise ImportError if module cannot be loaded
    module = importlib.import_module(name2model[model_name].ModuleName)

    # Read .JSON file according to the architecture to be loaded
    model_args_path = os.getenv('IFN_DIR_MODEL_CONFIGS') + model_name + ".json"                                         # json path
    assert os.path.isfile(model_args_path), "Cannot find file {}".format(model_args_path)                               # assert json path
    print("Loading model definitions from {} ".format(model_args_path) + "for model {}".format(model_name))
    model_args = Params(model_args_path)                                                                                # load model specificies

    # Get the class; will raise AttributeError if class cannot be found
    class_ = getattr(module, name2model[model_name].instantClass)

    # Load model
    load_function = str_to_class(name2model[model_name].modelDefLoad)                                                   # get the function to load the model
    model = load_function(class_, model_args)                                                                           # call the model-specific function to load the model definition
    print("Model definition based on", model_name, "was loaded.")

    return model


# Load state function (load weights)
def load_model_state(model, model_name, model_path):
    """Load model state checkpoint from disk"""

    # Load the module which includes model def; will raise ImportError if module cannot be loaded
    module = importlib.import_module(name2model[model_name].ModuleName)

    # Check weight path
    model_weight_path = model_path                                                                                     # weights path
    assert os.path.isfile(model_weight_path), "Cannot find weight file {}".format(model_weight_path)                  # assert weights path
    print("Loading model state from {} ".format(model_weight_path) + "for model {}".format(model_name))

    # Load model state
    load_function = str_to_class(name2model[model_name].modelStateLoad)                                                 # get the function to load the model
    model_ = load_function(model, module, model_weight_path)                                                                                   # call the model-specific function to load the model definition
    print("Model state based on", model_name, "was loaded.")

    return model_


def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    return model


# Main for testing
if __name__ == "__main__":
    # Print all the models
    print("List of efficient semantic segmentation models:")
    print("")
    print("    {:>14} | {:>3} | {:>30} | {:>14} | {:>70} | {:>50} | {:>50} |".format('name', 'id', 'ModuleName', 'instantClass', 'modelDefLoad', 'modelStateLoad'))
    print("    " + ('-' * 125))
    for model in models:
        print("    {:>14} | {:>3} | {:>30} | {:>14} | {:>70} | {:>50} | {:>50} |".format(model.name, model.id, model.ModuleName, model.instantClass, model.modelDefLoad, model.modelStateLoad))
    print("")
    print("Example usages:")

    # Map from name to model
    name = 'erfnet'
    id = name2model[name].id
    print("ID of model '{name}': {id}".format(name=name, id=id))

    # Map from ID to model
    ModuleName = id2model[id].ModuleName
    print("Name of Module of model with ID '{id}': {modulename}".format(id=id, modulename=ModuleName))

    # Example to load a model definition: load model definition based on 'erfnet':
    # The load_model_def calls the 'Load_model_erfnet()' function that loads a 'erfnet' model definition
    # model = load_model_def('name')
    model = load_model_def('erfnet')                                                                                    # load model definition

    # Load model state based on 'erfnet': loads the models weights  for the model 'erfnet
    weightspath = "/beegfs/work/shared/00_init_weights/semantic_segmentation/erfnet/erfnet_pretrained.pth"              # path to the weights to be loaded
    # model = load_model_state(model, 'name', weightspath)
    model = load_model_state(model, 'erfnet', weightspath)                                                              # load model state based on the weights on 'weightspath'