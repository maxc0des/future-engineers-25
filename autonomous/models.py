import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageNN(nn.Module):
    def __init__(self, output_size=128):  # Feature-Vektor-Größe für späteres Kombinieren
        super(ImageNN, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # ResNet-18 als Basis
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)  # Letzte Schicht anpassen

    def forward(self, x):
        return self.resnet(x)

class TOFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 128),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

class FullNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_nn = ImageNN()
        self.tof_nn = TOFNN()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128 + 128, 512),  
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  
        )

    def forward(self, image, tof):
        image_features = self.image_nn(image)
        tof_features = self.tof_nn(tof)
        combined = torch.cat((image_features, tof_features), dim=1)
        return self.linear_relu_stack(combined)
    
class IntegratedNN(nn.Module):
    def __init__(self, output_size=1): # ⚠️ OUTPUT-SIZE ÄNDERN ⚠️
        super(IntegratedNN, self).__init__()

        # --- Convolutional Layers ---
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 32 x 64 x 64
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 64 x 32 x 32
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 128 x 16 x 16
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 256 x 8 x 8
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 512 x 4 x 4
        )

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 512 x 4 x 4
        )

        self.conv_block7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 512 x 4 x 4
        )

        self.conv_block8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 512 x 4 x 4
        )

        self.conv_block9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 512 x 4 x 4
        )

        self.conv_block10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 512 x 4 x 4
        )

        self.conv_block11 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 512 x 4 x 4
        )

        # --- Adaptive Pooling, um Flatten-Größe flexibel zu halten ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # -> 256 x 4 x 4

        # --- Fully Connected Layers ---
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),  # Flattened size = 256 * 4 * 4 = 4096
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, output_size)  # Regression output (kein Activation)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        #print(x.shape)  # Vor dem Durchlauf durch conv_block4
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)
        #x = self.conv_block8(x)
        #x = self.conv_block9(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten für FC-Schichten

        x = self.fc_layers(x)
        return x
    
class EfficientNet5Channel(nn.Module):
    def __init__(self, output_size=2):
        super(EfficientNet5Channel, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)

        # Ersetze den ersten Conv2D Layer für 5 Kanäle
        old_conv = self.model.features[0][0]  # Zugriff auf den ersten Conv-Layer
        new_conv = nn.Conv2d(in_channels=5, out_channels=old_conv.out_channels, 
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
                             padding=old_conv.padding, bias=False)

        # Durchschnitt der RGB-Gewichte für die zusätzlichen 2 ToF-Kanäle
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight  # Übernehme RGB-Gewichte
            new_conv.weight[:, 3:] = old_conv.weight[:, :2].mean(dim=1, keepdim=True)  # Mittelwert für ToF

        # Ersetze den alten Layer mit dem neuen
        self.model.features[0][0] = new_conv

        # Ersetze die letzte Fully-Connected-Schicht für Regression (Steering & Velocity)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, output_size)

    def forward(self, x):
        return self.model(x)
    
class IntegratedNN2(nn.Module):
    def __init__(self):
        super(IntegratedNN2, self).__init__()

        # 🟢 Convolutional Layers mit BatchNorm
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=2)  # 5 Kanäle
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 96, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(96)
        self.conv5 = nn.Conv2d(96, 128, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(128)

        # 🟢 Fully Connected Layers mit Dropout
        self.fc1 = nn.Linear(128 * 1 * 18, 256)
        self.dropout1 = nn.Dropout(0.3)  # Verhindert Overfitting
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)  # 1 Output für den Lenkwinkel

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, start_dim=1)  # Flachmachen für FC-Layer
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Regressionsproblem, daher keine Aktivierung

        return x