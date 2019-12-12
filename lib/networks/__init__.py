from .resnet_cifar import *
from .resnet_cifar_analysis import *
from .resnext_cifar import *
from .wide_resnet_cifar import *
from .senet_pytorch import *
from .ncnet_pytorch import *
from .invcnn_pytorch import *
from .resnext import *
from .mobilenet_v2 import *

cifar_models = {"resnet20": resnet20a_cifar,
                "resnet56": resnet56a_cifar,
                "resnet110": resnet110a_cifar,
                "resnet164": resnet164_cifar,

                "resnet20a": resnet20a_cifar,
                "resnet56a": resnet56a_cifar,
                "resnet62a": resnet62a_cifar,
                "resnet68a": resnet68a_cifar,
                "resnet74a": resnet74a_cifar,
                "resnet80a": resnet80a_cifar,
                "resnet86a": resnet86a_cifar,
                "resnet92a": resnet92a_cifar,
                "resnet98a": resnet98a_cifar,
                "resnet104a": resnet104a_cifar,
                "resnet110a": resnet110a_cifar,

                "seresnet20": resnet20_cifar,
                "seresnet56": resnet56_cifar,
                "seresnet110": resnet110_cifar,
                "seresnet164": resnet164_cifar,

                "seresnet20a": resnet20a_cifar,
                "seresnet56a": resnet56a_cifar,
                "seresnet62a": resnet62a_cifar,
                "seresnet68a": resnet68a_cifar,
                "seresnet74a": resnet74a_cifar,
                "seresnet80a": resnet80a_cifar,
                "seresnet86a": resnet86a_cifar,
                "seresnet92a": resnet92a_cifar,
                "seresnet98a": resnet98a_cifar,
                "seresnet104a": resnet104a_cifar,
                "seresnet110a": resnet110a_cifar,

                "plainnet20": resnet20plain_cifar,
                "plainnet110": resnet110plain_cifar,

                "seplainnet20": resnet20plain_cifar,
                "seplainnet110": resnet110plain_cifar,

                "wresnet20": wresnet20_cifar,
                "sewresnet20": wresnet20_cifar,

                "resnext110": resneXt110_cifar,
                "seresnext110": resneXt110_cifar,

                "invcnn4": inv_cnn_4,
                }

imagenet_models = {"seresnet18": se_resnet18,
                   "seresnet50": se_resnet50,
                   "seresnet101": se_resnet101,
                   "ncresnet101": nc_resnet101,
                   "ncvgg16": nc_vgg16,
                   "resnext50": resnext50,
                   "seresnext50": seresnext50,
                   "ncresnext50": ncresnext50,
                   "mobilenetv2": MobileNetV2,
                   "semobilenetv2": SEMobileNetV2,
                   "ncmobilenetv2": NCMobileNetV2,
                   "sedensenet121": se_densenet121,
                   }
