# edm2-guidance

```
.
├── base/ # EDM2官方程式碼 基本上沒有做修改
├── classifier/ # OpenAI guided_diffusion官方程式碼 只是借用模型架構供MAP所需classifier架構以用來讀取訓練好的cl_9
├── edm2impl/ # 各種會用到的generate,evaluate script及Sampler的Parent class定義(Const CFG)
│   ├── calculate_is.py             # 計算Inception Score
│   ├── calculate_ms_ssim.py        # 計算MS SSIM
│   ├── fid_matrix_builder.py       # 用於建立FN DDG的Code
│   ├── generate_images_for_test.py # 主要生成圖片的Code
│   ├── generate_images.py          # 被廢棄的Code 請以generate_images_for_test.py為主
│   ├── plot_trajectories.py        # 用於繪製Denoise過程中的路徑圖
│   ├── prob_checking.py            # 用於儲存Prob Curve的變種generate_images_for_test.py
│   ├── utils.py                    # 部分額外輔助code
│   └── sampler_run.py              # Sampler的Parent class定義(Const CFG)
├── sampler/                        # DDG Sampler實作(不含Const CFG)
│   ├── guidances/                  # 各式Sampler的.py檔
│   ├── guidance.py                 # 用於import的.py
│   └── scheduler.py                # 所有Guidance Scale的 Scheduling
├── chen_main.ipynb                 # 少量張數視覺化
└── convert.ipynb                   # 呼叫EDM2 Code將Imagenet轉為訓練用的64*64*8 latent的code
````
