# Resnet50

```
.
├── base/ # EDM2官方程式碼 這裡只是借用1.多GPU操作Code,2.Dataset 
│   └── Resnet50/               #新版Classifier訓練 包含各式Classifier嘗試如
├── guided_diffusion/ # OpenAI guided_diffusion官方程式碼 只是借用模型架構供cl_9等使用
├── classifier_test_result/ # 視覺化訓練好的classifier在各time steps的能力
├── previous trial/ # 各種訓練失敗的classifier 可忽略
├── cl_x_*.py    # 第X版 Classifier訓練所使用的程式碼(用來給ipynb呼叫)
└── cl_x_*.ipynb # 第X版 Classifier訓練的呼叫指令(等價於使用terminal呼叫)
````
唯一需要注意的是cl_9_openai_62是最佳訓練結果，訓練在EDM2的latent空間，用於x0 prediction。但呼叫的api仍需要丟入Dummy t(torch.ones(labels_idx.shape[0]).to(device)))
訓練好的.pt檔額外上傳到雲端硬碟(請參考Root的Readme)
