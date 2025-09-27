def create_resnet50_classifier(num_classes, in_channels, pretrained,**other_kwargs):
    """
    根據參數創建和修改模型
    """
    # 載入模型
    model = models.resnet18(pretrained=pretrained)
    
    # 修改輸入通道數
    if in_channels != 3:
        original_conv1 = model.conv1
        new_conv1 = nn.Conv2d(in_channels, original_conv1.out_channels, 
                              kernel_size=original_conv1.kernel_size,
                              stride=original_conv1.stride,
                              padding=original_conv1.padding,
                              bias=False)
        
        # 如果使用預訓練權重，需要處理權重
        if pretrained:
            original_weights = original_conv1.weight.data
            new_conv1.weight.data[:, :3, :, :] = original_weights
            # 將原始權重的平均值複製到新的通道上
            if in_channels > 3:
                avg_weight = torch.mean(original_weights, dim=1, keepdim=True)
                new_conv1.weight.data[:, 3:in_channels, :, :] = avg_weight.expand(-1, in_channels - 3, -1, -1)

        model.conv1 = new_conv1

    # 修改最後一層全連接層
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),  # 增加一個中間層
        nn.ReLU(),
        nn.Dropout(0.5),              # 加入 Dropout 防止過度擬合
        nn.Linear(512, num_classes)   # 最後的全連接層
    )
        
    return model