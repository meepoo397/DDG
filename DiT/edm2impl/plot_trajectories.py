from base.torch_utils import distributed as dist
from utils import debug_print

def plot_all_trajectories(
    trajectories,
    trajectory_labels=None,
    mapping_path='./plot/class_labels.csv',
    scatter_csv='./plot/pca_scatter_points.csv',
    out_path='all_pca_trajectories.png',
    debug = False,
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.lines import Line2D

    # 載入 class mapping
    class_mapping = None
    try:
        df = pd.read_csv(mapping_path)
        class_mapping = dict(zip(df['Class ID'], df['Class Name']))
        dist.print0(f"[INFO] 載入 {len(class_mapping)} 筆 class label 映對")
    except Exception as e:
        dist.print0(f"[警告] 讀取 class_labels.csv 失敗: {e}")


    fig, ax = plt.subplots(figsize=(10, 8))

    # 1➖ 收集 trajectory labels
    traj_labels = sorted(set(trajectory_labels)) if trajectory_labels else []
    traj_labels = [str(l) for l in traj_labels]
    traj_labels_name = [class_mapping.get(str(l), f'class_{l}') if class_mapping else f'class_{l}' for l in trajectory_labels]
    cmap = cm.get_cmap('tab20', len(traj_labels))
    label_colors = dict(zip(traj_labels, cmap(range(len(traj_labels)))))

    # 2➖ 記錄 legend elements
    legend_elements = []
    seen_labels = set()
    

    # 4➖ 加入背景散點 (trajectory 的 class + 額外 top10)
    if scatter_csv and os.path.exists(scatter_csv):
        try:
            df = pd.read_csv(scatter_csv)
            # print(f'aaa:{df.columns}')#Index(['PC1', 'PC2', 'label', 'class_name'], dtype='object')
            # 額外 top 10 label (excludes trajectory labels)
            # top_labels = df['class_name'].value_counts().index.tolist()
            # extra_labels = [next((k for k, v in class_mapping.items() if v == l), l) for l in top_labels if l not in traj_labels_name][:10]
            # cmap_extra = cm.get_cmap('Set3', len(extra_labels))
            # print(f'ccc:{extra_labels}')
            # 合併 labels
            total_labels = traj_labels #+ extra_labels
            total_labels_name = [class_mapping.get(str(label), f'class_{label}') if class_mapping else f'class_{label}' for label in total_labels]
            df_target = df[df['class_name'].isin(total_labels_name)]
            # # 分配額外 class 顏色
            # for i, label in enumerate(extra_labels):
            #     print(f'eee:{label}')
            #     label_colors[str(label)] = cmap_extra(i)

            for label in total_labels:
                name = class_mapping.get(str(label), f'class_{label}') if class_mapping else f'class_{label}'
                subset = df_target[df_target['class_name'] == name]
                if subset.empty:
                    continue
                color = label_colors[str(label)]
                class_name = subset['class_name'].iloc[0] if 'class_name' in subset.columns else f'class_{label}'
                display_name = class_name.split(',')[0][:20]
                ax.scatter(
                    subset['PC1'], subset['PC2'],
                    label=f"(Ground Truth) {display_name}",
                    color=color,
                    alpha=0.7,
                    s=20
                )

            debug_print(debug,f"加入背景散點：{len(df_target)} 點, 類別數：{len(total_labels)})")
        except Exception as e:
            debug_print(debug,f"載入 {scatter_csv} 時錯誤: {e}")
    # 3➖ 畫 trajectory
    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        x, y = traj[:-1, 0], traj[:-1, 1]
        dx, dy = traj[1:, 0] - x, traj[1:, 1] - y

        if trajectory_labels is not None:
            label = trajectory_labels[i]
            debug_print(debug,f'Using label={str(label)}--- to search in class_mapping')
            # print(f'class mapping:{class_mapping}')
            debug_print(debug,f'Found:{class_mapping[str(label)] if class_mapping else f"class_{label}"}')
            debug_print(debug,f'label_colors:{label_colors}')
            color = label_colors[str(label)]
            debug_print(debug,f'Finding:{str(label)}')
            class_name = class_mapping.get(str(label), f'class_{label}') if class_mapping else f'class_{label}'
            display_name = class_name.split(',')[0][:20]

            ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1,
                      width=0.003, color=color, alpha=0.9)

            if label not in seen_labels:
                legend_elements.append(Line2D([0], [0], color=color, lw=2, label=display_name))
                seen_labels.add(label)
        else:
            ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1,
                      width=0.003, color='blue', alpha=0.7)

    # 5➖ legend 設定
    handles_bg, labels_bg = ax.get_legend_handles_labels()
    ax.legend(legend_elements + handles_bg, 
            [h.get_label() for h in legend_elements] + labels_bg, 
            fontsize=8, loc='best')
    # 6➖ 其他圖設
    ax.set_title("PCA Trajectories of All Images")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()