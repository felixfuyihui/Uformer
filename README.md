# Uformer
The implementation of Uformer: A Unet based dilated complex &amp; real dual-path conformer network for simultaneous speech enhancement and dereverberation

The paper is available at: https://arxiv.org/abs/2111.06015

Please cite the paper if you want to follow the idea or results:

    @article{fu2021uformer,
            title={Uformer: A Unet based dilated complex \& real dual-path conformer network for simultaneous speech enhancement and dereverberation},
            author={Fu, Yihui and Liu, Yun and Li, Jingdong and Luo, Dawei and Lv, Shubo and Jv, Yukai and Xie, Lei},
            journal={arXiv preprint arXiv:2111.06015},
            year={2021}
            }

2022.6.18
I make some modifications of the model including:
1. Replace local temporal attention with global temporal attention.
2. Cancel Encoder Decoder Attention to reduce the amount of parameter.
3. Change some activation function and norm function.
