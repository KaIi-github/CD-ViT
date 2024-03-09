# CD-ViT

This project is based on the official code of the paper "Contextual Dependency Vision Transformer for Spectrogram-based Multivariate Time Series Analysis".

## Environment Setup

Install the required dependencies using the following command:
   ```
   conda create -n cd_vit python=3.8
   source activate cd_vit
   pip install torch==1.10.1 torchvision==0.11.2 
   pip install seaborn==0.12.2 timm==0.4.12 scikit-learn==1.3.2 tqdm==4.64.1
   ```

## How to Run

Execute the following command in the command line to start the program:
   ```
   python main.py
   ```
## Citation

Please cite CD-ViT in your publications if it helps your research. The following is a BibTeX for our paper.

	@article{yao2024contextual,
	  title={Contextual Dependency Vision Transformer for Spectrogram-based Multivariate Time Series Analysis},
	  author={Yao, Jieru and Han, Longfei and Yang, Kaihui and Guo, Guangyu and Liu, Nian and Huang, Xiankai and Zheng, Zhaohui and Zhang, Dingwen and Han, Junwei},
	  journal={Neurocomputing},
	  volume={572},
	  pages={127215},
	  year={2024},
	  publisher={Elsevier}
	}
	
