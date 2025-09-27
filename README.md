# DDG

請注意有部分大型檔案須經由以下雲端硬碟連結下載
[雲端硬碟連結](https://drive.google.com/drive/folders/1s-PDCZiWCf6Qs9FYGn3aV3jyfxZr8zRl?usp=sharing)
```
.
├── Classifier/ # Classifier訓練相關Code
│   ├── guided-diffusion-mp/    #舊版Classifier訓練 最單純的使用OpenAI在guided-diffusion提供的架構 單純修改time embedding為EDM2的embedding(最終放棄使用)
│   └── Resnet50/               #新版Classifier訓練 包含各式Classifier嘗試如Resnet,ViT,x0 prediction版本
├── DiT/ # 實作Scaled difference schedule 在DiT時的Code
└── edm2-guidance # 所有跟DDG Sampling實作及生成圖片相關的code 基於edm2實作
````
