from torchvision import models
from torch import nn


def load_model(name, config):
    
    if name == 'ResNet18_v1.0':
        try:
            l1=config["l1"]
            l2=config["l2"]
            dropout=config['dropout']
        except:
            print("Initializing the model failed. Check whether config contains all necessary variables for {}.".format(name))
            exit()
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(512, l1),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l1, l2),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l2, 5),
                                 nn.Sigmoid(),
        )
        return model
    
    # Unfreeze and re-train Layer4 of ResNet18
    if name == 'ResNet18_v2.0':
        try:
            l1=config["l1"]
            l2=config["l2"]
            dropout=config['dropout']
        except:
            print("Initializing the model failed. Check whether config contains all necessary variables for {}.".format(name))
            exit()
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        model.fc = nn.Sequential(nn.Linear(512, l1),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l1, l2),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l2, 5),
                                 nn.Sigmoid(),
        )
        return model
    

    # Unfreeze and re-train Layer1-4 of ResNet18
    if name == 'ResNet18_v3.0':
        try:
            l1=config["l1"]
            l2=config["l2"]
            dropout=config['dropout']
        except:
            print("Initializing the model failed. Check whether config contains all necessary variables for {}.".format(name))
            exit()
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = True
        for param in model.layer2.parameters():
            param.requires_grad = True
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        model.fc = nn.Sequential(nn.Linear(512, l1),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l1, l2),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l2, 5),
                                 nn.Sigmoid(),
        )
        return model

    
    if name == 'ResNet34_v1.0':
        try:
            l1=config["l1"]
            l2=config["l2"]
            dropout=config['dropout']
        except:
            print("Initializing the model failed. Check whether config contains all necessary variables for {}.".format(name))
            exit()
        model = models.resnet34(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(512, l1),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l1, l2),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l2, 5),
                                 nn.Sigmoid(),
        )
        return model
    
    
    if name == 'ResNet34_v2.0':
        try:
            dropout=config['dropout']
        except:
            print("Initializing the model failed. Check whether config contains all necessary variables for {}.".format(name))
            exit()
        model = models.resnet34(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(512, 5),
                                 nn.Sigmoid(),
        )
        return model
    
    

    
    if name == 'ResNet50_v1.0':
        try:
            l1=config["l1"]
            l2=config["l2"]
            dropout=config['dropout']
        except:
            print("Initializing the model failed. Check whether config contains all necessary variables for {}.".format(name))
            exit()
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(2048, l1),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l1, l2),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l2, 5),
                                 nn.Sigmoid(),
        )
        return model
    
    
    if name == 'ResNet101_v1.0':
        try:
            l1=config["l1"]
            l2=config["l2"]
            l3=config["l3"]
            dropout=config['dropout']
        except:
            print("Initializing the model failed. Check whether config contains all necessary variables for {}.".format(name))
            exit()
        model = models.resnet101(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(2048, l1),
                         nn.ReLU(True),
                         nn.Dropout(p=dropout),
                         nn.Linear(l1, l2),
                         nn.ReLU(True),
                         nn.Dropout(p=dropout),
                         nn.Linear(l2, l3),
                         nn.ReLU(True),
                         nn.Dropout(p=dropout),
                         nn.Linear(l3, 5),
                         nn.Sigmoid(),
                                )
        return model
    
    
    if name == 'ResNet101_v2.0':
        try:
            pass
        except:
            print("Initializing the model failed. Check whether config contains all necessary variables for {}.".format(name))
            exit()
        model = models.resnet101(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(True),
                         nn.Dropout(p=0.5),
                         nn.Linear(512, 256),
                         nn.ReLU(True),
                         nn.Dropout(p=0.3),
                         nn.Linear(256, 64),
                         nn.ReLU(True),
                         nn.Dropout(p=0.1),
                         nn.Linear(64, 5),
                         nn.Sigmoid(),
                                )
        return model


    
    if name == 'ResNet152_v1.0':
        try:
            l1=config["l1"]
            l2=config["l2"]
            dropout=config['dropout']
        except:
            print("Initializing the model failed. Check whether config contains all necessary variables for {}.".format(name))
            exit()
        model = models.resnet152(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(2048, l1),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l1, l2),
                                 nn.ReLU(True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(l2, 5),
                                 nn.Sigmoid(),
        )
        return model

    