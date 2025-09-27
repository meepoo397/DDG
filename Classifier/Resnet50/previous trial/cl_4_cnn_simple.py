def create_resnet50_classifier(num_classes, in_channels, pretrained,**other_kwargs):
    class VAEClassifier(nn.Module):
        def __init__(self, num_classes=1000,in_channels=4):
            super().__init__()
    
            self.features = nn.Sequential(
                # 第一層: 將 4 個通道轉換為 32 個通道
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32
    
                # 第二層: 增加通道數到 64
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
    
                # 第三層: 增加通道數到 128
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
    
                # 第四層: 增加通道數到 256
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
            )
    
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5), # 增加 Dropout 以防止過度擬合
                nn.Linear(256 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )
    
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    return VAEClassifier(num_classes=num_classes,in_channels=in_channels)