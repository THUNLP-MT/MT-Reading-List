# Machine Translation Reading List
This is a machine translation reading list maintained by the Tsinghua Natural Language Processing Group.

* [Statistical Machine Translation](#statistical_machine_translation)
    * [Word-based Models](#word_based_models)
    * [Phrase-based Models](#phrase_based_models)
    * [Syntax-based Models](#phrase_based_models)
    * [Discriminative Training](#discriminative_training)
    * [System Combination](#system_combination)
    * [Evaluation](#evaluation)
 * [Neural Machine Translation](#neural_machine_translation)
    * [Model Architecture](#model_architecture)
    * [Attention Mechanism](#attention_mechanism)
    * [Open Vocabulary](#open_vocabulary)
    * [Training](#training)
    * [Decoding](#decoding)
    * [Low-resource Language Translation](#low_resource_language_translation)
    * [Multiple Language Translation](#multiple_language_translation)
    * [Prior Knowledge Integration](#prior_knowledge_integration)
    * [Document-level Translation](#document_level_translation)
    * [Robustness](#robustness)
    * [Visualization and Interpretability](#visualization_and_interpretability)
    * [Efficiency](#efficiency)
    * [Multi-modality](#multi_modality)
    * [Domain Adaptation](#domain_adaptation)
    * [Quality Estimation](#quality_estimation)

<h2 id="statistical_machine_translation">Statistical Machine Translation</h2>

<h3 id="word_based_models">Word-based Models</h3>

* Peter E. Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, and Robert L. Mercer. 1993. [The Mathematics of Statistical Machine Translation: Parameter Estimation](http://aclweb.org/anthology/J93-2003). *Computational Linguistics*.
* Stephan Vogel, Hermann Ney, and Christoph Tillmann. 1996. [HMM-Based Word Alignment in Statistical Translation](http://aclweb.org/anthology/C96-2141). In *Proceedings of COLING 1996*.
* Franz Josef Och and Hermann Ney. 2002. [Discriminative Training and Maximum Entropy Models for Statistical Machine Translation](http://aclweb.org/anthology/P02-1038). *Computational Linguistics*.
* Franz Josef Och and Hermann Ney. 2003. [A Systematic Comparison of Various Statistical Alignment Models](http://aclweb.org/anthology/J03-1002). *Computational Linguistics*.

<h3 id="phrase_based_models">Phrase-based Models</h3>

* Philipp Koehn, Franz J. Och, and Daniel Marcu. 2003. [Statistical Phrase-Based Translation](http://aclweb.org/anthology/N03-1017). In *Proceedings of NAACL 2003*.

<h3 id="syntax_based_models">Syntax-based Models</h3>

* Dekai Wu. 1997. [Stochastic Inversion Transduction Grammars and Bilingual Parsing of Parallel Corpora](http://aclweb.org/anthology/J97-3002). *Computational Linguistics*.
* Michel Galley, Jonathan Graehl, Kevin Knight, Daniel Marcu, Steve DeNeefe, Wei Wang, and Ignacio Thayer. 2006. [Scalable Inference and Training of Context-Rich Syntactic Translation Models](http://aclweb.org/anthology/P06-1121). In *Proceedings of ACL 2006*
* Yang Liu, Qun Liu, and Shouxun Lin. 2006. [Tree-to-String Alignment Template for Statistical Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/P06-1077.pdf). In *Proceedings of COLING/ACL 2006*.
* David Chiang. 2007. [Hierarchical Phrase-Based Translation](http://aclweb.org/anthology/J07-2003). *Computational Linguistics*.
* Libin Shen, Jinxi Xu, and Ralph Weischedel. 2008. [A New String-to-Dependency Machine Translation Algorithm with a Target Dependency Language Model](http://aclweb.org/anthology/P08-1066). In *Proceedings of ACL 2008*.

<h3 id="discriminative_training">Discriminative Training</h3>

* Franz Josef Och. 2003. [Minimum Error Rate Training in Statistical Machine Translation](http://aclweb.org/anthology/P03-1021). In *Proceedings of ACL 2003*.
* Taro Watanabe, Jun Suzuki, Hajime Tsukada, and Hideki Isozaki. 2007. [Online Large-Margin Training for Statistical Machine Translation](http://aclweb.org/anthology/D07-1080). In *Proceedings of EMNLP 2007*.
* David Chiang, Kevin Knight, and Wei Wang. 2009. [11,001 New Features for Statistical Machine Translation](http://aclweb.org/anthology/N09-1025). In *Proceedings of NAACL 2009*.

<h3 id="system_combination">System Combination</h3>

* Antti-Veikko Rosti, Spyros Matsoukas, and Richard Schwartz. 2007. [Improved Word-Level System Combination for Machine Translation](http://aclweb.org/anthology/P07-1040). In *Proceedings of ACL 2007*.
* Xiaodong He, Mei Yang, Jianfeng Gao, Patrick Nguyen, and Robert Moore. 2008. [Indirect-HMM-based Hypothesis Alignment for Combining Outputs from Machine Translation Systems](http://aclweb.org/anthology/D08-1011). In *Proceedings of EMNLP 2008*.

<h3 id="evaluation">Evaluation</h3>

* Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. [BLEU: a Method for Automatic Evaluation of Machine Translation](http://aclweb.org/anthology/P02-1040). In *Proceedings of ACL 2002*.

<h2 id="neural_machine_translation">Neural Machine Translation</h2>

<h3 id="model_architecture">Model Architecture</h3>

* Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. [Sequence to Sequence Learning
with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf). In *Proceedings of NIPS 2014*.
* Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473). In *Proceedings of ICLR 2015*.
* Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2016. [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144). *arXiv:1609.08144*.
* Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. 2017. [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf). *arXiv:1705.03122*.
* Jonas Gehring, Michael Auli, David Grangier, and Yann Dauphin. 2017. [A Convolutional Encoder Model for Neural Machine Translation](http://aclweb.org/anthology/P17-1012). In *Proceedings of ACL 2017*.
* Mingxuan Wang, Zhengdong Lu, Jie Zhou, and Qun Liu. 2017. [Deep Neural Machine Translation with Linear Associative Unit](http://aclweb.org/anthology/P17-1013). In *Proceedings of ACL 2017*.
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. [Attention is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). In *Proceedings of NIPS 2017*.
* Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Mike Schuster, Noam Shazeer, Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Zhifeng Chen, Yonghui Wu, and Macduff Hughes. 2018. [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation](http://aclweb.org/anthology/P18-1008). In *Proceedings of ACL 2018*.
* Zi-Yi Dou, Zhaopeng Tu, Xing Wang, Shuming Shi, and Tong Zhang. 2018. [Exploiting Deep Representations for Neural Machine Translation](http://aclweb.org/anthology/D18-1457). In *Proceedings of EMNLP 2018*.
* Biao Zhang, Deyi Xiong, Jinsong Su, Qian Lin, and Huiji Zhang. 2018. [Simplifying Neural Machine Translation with Addition-Subtraction Twin-Gated Recurrent Networks](http://aclweb.org/anthology/D18-1459). In *Proceedings of EMNLP 2018*.
* Gongbo Tang, Mathias Müller, Annette Rios, and Rico Sennrich. 2018. [Why Self-Attention? A Targeted Evaluation of Neural Machine Translation Architectures](http://aclweb.org/anthology/D18-1458). In *Proceedings of EMNLP 2018*.

<h3 id="attention_mechanism">Attention Mechanism</h3>

* Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025). In *Proceedings of EMNLP 2015*.
* Tao Shen, Tianyi Zhou, Guodong Long, Jing Jiang, Shirui Pan, and Chengqi Zhang. 2018. [DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding](https://arxiv.org/pdf/1709.04696.pdf). In *Proceedings of AAAI 2018*.
* Peter Shaw, Jakob Uszkorei, and Ashish Vaswani. 2018. [Self-Attention with Relative Position Representations](http://aclweb.org/anthology/N18-2074). In *Proceedings of NAACL 2018*.
* Biao Zhang, Deyi Xiong, and Jinsong Su. 2018. [Accelerating Neural Transformer via an Average Attention Network](http://aclweb.org/anthology/P18-1166). In *Proceedings of ACL 2018*.
* Tobias Domhan. 2018. [How Much Attention Do You Need? A Granular Analysis of Neural Machine Translation Architectures](http://aclweb.org/anthology/P18-1167). In *Proceedings of ACL 2018*.
* Shaohui Kuang, Junhui Li, António Branco, Weihua Luo, and Deyi Xiong. 2018. [Attention Focusing for Neural Machine Translation by Bridging Source and Target Embeddings](http://aclweb.org/anthology/P18-1164). In *Proceedings of ACL 2018*.
* Jian Li, Zhaopeng Tu, Baosong Yang, Michael R. Lyu, and Tong Zhang. 2018. [Multi-Head Attention with Disagreement Regularization](http://aclweb.org/anthology/D18-1317). In *Proceedings of EMNLP 2018*.

<h3 id="open_vocabulary">Open Vocabulary</h3>

* Thang Luong, Ilya Sutskever, Quoc Le, Oriol Vinyals, and Wojciech Zaremba. 2015. [Addressing the Rare Word Problem in Neural Machine Translation](http://aclweb.org/anthology/P15-1002). In *Proceedings of ACL 2015*.
* Sébastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. 2015. [On Using Very Large Target Vocabulary for Neural Machine Translation](http://www.aclweb.org/anthology/P15-1001). In *Proceedings of ACL 2015*.
* Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf). In *Proceedings of ACL 2016*.
* Yonatan Belinkov, Nadir Durrani, Fahim Dalvi, Hassan Sajjad, and James Glass. 2017. [What do Neural Machine Translation Models Learn about Morphology?](http://aclweb.org/anthology/P17-1080). In *Proceedings of ACL 2017*.
* Taku Kudo. 2018. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](http://aclweb.org/anthology/P18-1007). In *Proceedings of ACL 2018*.
* Yang Zhao, Jiajun Zhang, Zhongjun He, Chengqing Zong, and Hua Wu. 2018. [Addressing Troublesome Words in Neural Machine Translation](http://aclweb.org/anthology/D18-1036). In *Proceedings of EMNLP 2018*.
* Colin Cherry, George Foster, Ankur Bapna, Orhan Firat, and Wolfgang Macherey. 2018. [Revisiting Character-Based Neural Machine Translation with Capacity and Compression](http://aclweb.org/anthology/D18-1461). In *Proceedings of EMNLP 2018*.
* Arianna Bisazza and Clara Tump. 2018. [The Lazy Encoder: A Fine-Grained Analysis of the Role of Morphology in Neural Machine Translation](http://aclweb.org/anthology/D18-1313). In *Proceedings of EMNLP 2018*.

<h3 id="training">Training</h3>

* Shiqi Shen, Yong Cheng, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. [Minimum Risk Training for Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_mrt.pdf). In *Proceedings of ACL 2016*.
* Sam Wiseman and Alexander M. Rush. 2016. [Sequence-to-Sequence Learning as Beam-Search Optimization](http://aclweb.org/anthology/D16-1137). In *Proceedings of EMNLP 2016*.

<h3 id="decoding">Decoding</h3>

* Shonosuke Ishiwatari, Jingtao Yao, Shujie Liu, Mu Li, Ming Zhou, Naoki Yoshinaga, Masaru Kitsuregawa, and Weijia Jia. 2017. [Chunk-based Decoder for Neural Machine Translation](http://aclweb.org/anthology/P17-1174). In *Proceedings of ACL 2017*.
* Philip Schulz, Wilker Aziz, and Trevor Cohn. 2018. [A Stochastic Decoder for Neural Machine Translation](http://aclweb.org/anthology/P18-1115). In *Proceedings of ACL 2018*.
* Jiatao Gu, James Bradbury, Caiming Xiong, Victor O.K. Li, and Richard Socher. 2018. [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281). In *Proceedings of ICLR 2018*.
* Chunqi Wang, Ji Zhang, and Haiqing Chen. 2018. [Semi-Autoregressive Neural Machine Translation](http://aclweb.org/anthology/D18-1044). In *Proceedings of EMNLP 2018*.
* Xiangwen Zhang, Jinsong Su, Yue Qin, Yang Liu, Rongrong Ji, and Hongji Wang. 2018. [Asynchronous Bidirectional Decoding for Neural Machine Translation](https://arxiv.org/pdf/1801.05122). In *Proceedings of AAAI 2018*.
* Xinwei Geng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2018. [Adaptive Multi-pass Decoder for Neural Machine Translation](http://aclweb.org/anthology/D18-1048). In *Proceedings of EMNLP 2018*.
* Wen Zhang, Liang Huang, Yang Feng, Lei Shen, and Qun Liu. 2018. [Speeding Up Neural Machine Translation Decoding by Cube Pruning](http://aclweb.org/anthology/D18-1460). In *Proceedings of EMNLP 2018*.
* Xinyi Wang, Hieu Pham, Pengcheng Yin, and Graham Neubig. 2018. [A Tree-based Decoder for Neural Machine Translation](http://aclweb.org/anthology/D18-1509). In *Proceedings of EMNLP 2018*.
* Chenze Shao, Xilin Chen, and Yang Feng. 2018. [Greedy Search with Probabilistic N-gram Matching for Neural Machine Translation](http://aclweb.org/anthology/D18-1510). In *Proceedings of EMNLP 2018*.
* Zhisong Zhang, Rui Wang, Masao Utiyama, Eiichiro Sumita, and Hai Zhao. 2018. [Exploring Recombination for Efficient Decoding of Neural Machine Translation](http://aclweb.org/anthology/D18-1511). In *Proceedings of EMNLP 2018*.
* Jetic Gū, Hassan S. Shavarani, and Anoop Sarkar. 2018. [Top-down Tree Structured Decoding with Syntactic Connections for Neural Machine Translation and Parsing](http://aclweb.org/anthology/D18-1037). In *Proceedings of EMNLP 2018*.

<h3 id="low_resource_language_translation">Low-resource Language Translation</h3>

* Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. [Improving Neural Machine Translation Models with Monolingual Data](https://arxiv.org/pdf/1511.06709). In *Proceedings of ACL 2016*.
* Yong Cheng, Wei Xu, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. [Semi-Supervised Learning for Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_semi.pdf). In *Proceedings of ACL 2016*.
* Barret Zoph, Deniz Yuret, Jonathan May, and Kevin Knight. 2016. [Transfer Learning for Low-Resource Neural Machine Translation](https://www.isi.edu/natural-language/mt/emnlp16-transfer.pdf). In *Proceedings of EMNLP 2016*.
* Di He, Yingce Xia, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, Wei-Ying Ma. 2016. [Dual Learning for Machine Translation](https://papers.nips.cc/paper/6469-dual-learning-for-machine-translation.pdf). In *Proceedings of NIPS 2016*.
* Marzieh Fadaee, Arianna Bisazza and Christof Monz. 2017. [Data Augmentation for Low-Resource Neural Machine Translation](http://aclweb.org/anthology/P/P17/P17-2090.pdf). In *Proceedings of ACL 2017*.
* Marlies van der Wees, Arianna Bisazza and Christof Monz. 2017. [Dynamic Data Selection for Neural Machine Translation](http://aclweb.org/anthology/D/D17/D17-1147.pdf). In *Proceedings of EMNLP 2017*.
* Yun Chen, Yang Liu, Yong Cheng and Victor O.K. Li. 2017. [A Teacher-Student Framework for Zero-resource Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_cy.pdf). In *Proceedings of ACL 2017*.
* Hao Zheng, Yong Cheng, and Yang Liu. 2017. [Maximum Expected Likelihood Estimation for Zero-resource Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/ijcai2017_zh.pdf). In *Proceedings of IJCAI 2017*.
* Yong Cheng, Qian Yang, Yang Liu, Maosong Sun, and Wei Xu. 2017. [Joint Training for Pivot-based Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/ijcai2017_cy.pdf). In *Proceedings of IJCAI 2017*.
* Zhen Yang, Wei Chen, Feng Wang, and Bo Xu. 2018. [Unsupervised Neural Machine Translation with Weight Sharing](http://aclweb.org/anthology/P18-1005). In *Proceedings of ACL 2018*.
* Shuo Ren, Wenhu Chen, Shujie Liu, Mu Li, Ming Zhou, and Shuai Ma. 2018. [Triangular Architecture for Rare Language Translation](http://aclweb.org/anthology/P18-1006). In *Proceedings of ACL 2018*.
* Yun Chen, Yang Liu, and Victor O. K. Li. 2018. [Zero-Resource Neural Machine Translation with Multi-Agent Communication Game](https://arxiv.org/pdf/1802.03116). In *Proceedings of AAAI 2018*.
* Marzieh Fadaee and Christof Monz. 2018. [Back-Translation Sampling by Targeting Difficult Words in Neural Machine Translation](https://aclanthology.coli.uni-saarland.de/events/emnlp-2018). In *Proceedings of EMNLP 2018*.
* Sergey Edunov, Myle Ott, Michael Auli, and David Grangier. 2018. [Understanding Back-Translation at Scale](http://aclweb.org/anthology/D18-1045). In *Proceedings of EMNLP 2018*.
* Guillaume Lample, Myle Ott, Alexis Conneau, Ludovic Denoyer, and Marc'Aurelio Ranzato. 2018. [Phrase-Based & Neural Unsupervised Machine Translation](http://aclweb.org/anthology/D18-1549). In *Proceedings of EMNLP 2018*.
* Xinyi Wang, Hieu Pham, Zihang Dai, and Graham Neubig. 2018. [SwitchOut: an Efficient Data Augmentation Algorithm for Neural Machine Translation](http://aclweb.org/anthology/D18-1100). In *Proceedings of EMNLP 2018*.

<h3 id="multiple_language_translation">Multiple Language Translation</h3>

* Daxiang Dong, Hua Wu, Wei He, Dianhai Yu, and Haifeng Wang. 2015. [Multi-Task Learning for Multiple Language Translation](http://aclweb.org/anthology/P15-1166). In *Proceedings of ACL 2015*.
* Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda Viégas, Martin Wattenberg, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2017. [Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://arxiv.org/pdf/1611.04558). *Transactions of the Association for Computational Linguistics*.
* Emmanouil Antonios Platanios, Mrinmaya Sachan, Graham Neubig, and Tom Mitchell. 2018. [Contextual Parameter Generation for
Universal Neural Machine Translation](http://aclweb.org/anthology/D18-1039). In *Proceedings of EMNLP 2018*.
* Yining Wang, Jiajun Zhang, Feifei Zhai, Jingfang Xu, and Chengqing Zong. 2018. [Three Strategies to Improve One-to-Many Multilingual Translation](http://aclweb.org/anthology/D18-1326). In *Proceedings of EMNLP 2018*.

<h3 id="prior_knowledge_integration">Prior Knowledge Integration</h3>

* Trevor Cohn, Cong Duy Vu Hoang, Ekaterina Vymolova, Kaisheng Yao, Chris Dyer, and Gholamreza Haffari. 2016. [Incorporating Structural Alignment Biases into an Attentional Neural Translation Model](https://arxiv.org/pdf/1601.01085.pdf). In *Proceedings of NAACL 2016*.
* Zhaopeng Tu, Zhengdong Lu, Yang Liu, Xiaohua Liu, and Hang Li. 2016. [Modeling Coverage for Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_coverage.pdf). In *Proceedings of ACL 2016*.
Yong Cheng, Shiqi Shen, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. [Agreement-based Joint Training for Bidirectional Attention-based Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/ijcai16_agree.pdf). In *Proceedings of IJCAI 2016*.
* Jiacheng Zhang, Yang Liu, Huanbo Luan, Jingfang Xu and Maosong Sun. 2017. [Prior Knowledge Integration for Neural Machine Translation using Posterior Regularization](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_zjc.pdf). In *Proceedings of ACL 2017*.
* Junhui Li, Deyi Xiong, Zhaopeng Tu, Muhua Zhu, Min Zhang, and Guodong Zhou. 2017. [Modeling Source Syntax for Neural Machine Translation](http://aclweb.org/anthology/P17-1064). In *Proceedings of ACL 2017*.
* Shuangzhi Wu, Dongdong Zhang, Nan Yang, Mu Li, and Ming Zhou. 2017. [Sequence-to-Dependency Neural Machine Translation](http://aclweb.org/anthology/P17-1065). In *Proceedings of ACL 2017*.
* Jinchao Zhang, Mingxuan Wang, Qun Liu, and Jie Zhou. 2017. [Incorporating Word Reordering Knowledge into Attention-based Neural Machine Translation](http://aclweb.org/anthology/P17-1140). In *Proceedings of ACL 2017*.
* Huadong Chen, Shujian Huang, David Chiang, and Jiajun Chen. 2017. [Improved Neural Machine Translation with a Syntax-Aware Encoder and Decoder](http://aclweb.org/anthology/P17-1177). In *Proceedings of ACL 2017*.
* Chris Hokamp and Qun Liu. 2017. [Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search](http://aclweb.org/anthology/P17-1141). In *Proceedings of ACL 2017*.
* Chunpeng Ma, Akihiro Tamura, Masao Utiyama, Tiejun Zhao, and Eiichiro Sumita. 2018. [Forest-Based Neural Machine Translation](http://aclweb.org/anthology/P18-1116). In *Proceedings of ACL 2018*.
* Anna Currey and Kenneth Heafield. 2018. [Multi-Source Syntactic Neural Machine Translation](http://aclweb.org/anthology/D18-1327). In *Proceedings of EMNLP 2018*.

<h3 id="document_level_translation">Document-level Translation</h3>

* Elena Voita, Pavel Serdyukov, Rico Sennrich, and Ivan Titov. 2018. [Context-Aware Neural Machine Translation Learns Anaphora Resolution](http://aclweb.org/anthology/P18-1117). In *Proceedings of ACL 2018*.
* Sameen Maruf and Gholamreza Haffari. 2018. [Document Context Neural Machine Translation with Memory Networks](http://aclweb.org/anthology/P18-1118). 2018. In *Proceedings of ACL 2018*.
* Jiacheng Zhang, Huanbo Luan, Maosong Sun, Feifei Zhai, Jingfang Xu, Min Zhang and Yang Liu. 2018. [Improving the Transformer Translation Model with Document-Level Context](http://aclweb.org/anthology/D18-1049). In *Proceedings of EMNLP 2018*.
* Zhaopeng Tu, Yang Liu, Shumin Shi, and Tong Zhang. 2018. [Learning to Remember Translation History with a Continuous Cache](https://arxiv.org/pdf/1711.09367.pdf). *Transactions of the Association for Computational Linguistics*.
* Samuel Läubli, Rico Sennrich, and Martin Volk. 2018. [Has Machine Translation Achieved Human Parity? A Case for Document-level Evaluation](http://aclweb.org/anthology/D18-1512). In *Proceedings of EMNLP 2018*.
* Lesly Miculicich, Dhananjay Ram, Nikolaos Pappas, and James Henderson. 2018. [Document-Level Neural Machine Translation with Hierarchical Attention Networks](http://aclweb.org/anthology/D18-1325). In *Proceedings of EMNLP 2018*.

<h3 id="robustness">Robustness</h3>

* Yong Cheng, Zhaopeng Tu, Fandong Meng, Junjie Zhai, and Yang Liu. 2018. [Towards Robust Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2018_cy.pdf). In *Proceedings of ACL 2018*.
* Paul Michel and Graham Neubig. 2018. [MTNT: A Testbed for Machine Translation of Noisy Text](http://aclweb.org/anthology/D18-1050). In *Proceedings of EMNLP 2018*.

<h3 id="visualization_and_interpretability">Visualization and Interpretability</h3>

* Yanzhuo Ding, Yang Liu, Huanbo Luan and Maosong Sun. 2017. [Visualizing and Understanding Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_dyz.pdf). In *Proceedings of ACL 2017*.

<h3 id="efficiency">Efficiency</h3>

* Abigail See, Minh-Thang Luong, and Christopher D. Manning. 2016. [Compression of Neural Machine Translation Models via Pruning](http://aclweb.org/anthology/K16-1029). In *Proceedings of CoNLL 2016*.
* Yusuke Oda, Philip Arthur, Graham Neubig, Koichiro Yoshino, and Satoshi Nakamura. 2017. [Neural Machine Translation via Binary Code Prediction](http://aclweb.org/anthology/P17-1079). In *Proceedings of ACL 2017*.
* Joern Wuebker, Patrick Simianer, and John DeNero. 2018. [Compact Personalized Models for Neural Machine Translation](http://aclweb.org/anthology/D18-1104). In *Proceedings of EMNLP 2018*.

<h3 id="domain_adaptation">Domain Adaptation</h3>

* Jiali Zeng, Jinsong Su, Huating Wen, Yang Liu, Jun Xie, Yongjing Yin, and Jianqiang Zhao. 2018. [Multi-Domain Neural Machine Translation with Word-Level Domain Context Discrimination](http://aclweb.org/anthology/D18-1041). In *Proceedings of EMNLP 2018*.
* Graham Neubig and Junjie Hu. 2018. [Rapid Adaptation of Neural Machine Translation to New Languages](http://aclweb.org/anthology/D18-1103). In *Proceedings of EMNLP 2018*.

<h3 id="multi_modality">Multi-modality</h3>

* Iacer Calixto, Qun Liu, and Nick Campbell. 2017. [Doubly-Attentive Decoder for Multi-modal Neural Machine Translation](http://aclweb.org/anthology/P17-1175). In *Proceedings of ACL 2017*.
* Desmond Elliott. 2018. [Adversarial Evaluation of Multimodal Machine Translation](http://aclweb.org/anthology/D18-1329). In *Proceedings of EMNLP 2018*.

<h3 id="quality_estimation">Quality Estimation</h3>

* Kai Fan, Jiayi Wang, Bo Li, Fengming Zhou, Boxing Chen, and Luo Si. 2019. ["Bilingual Expert" Can Find Translation Errors](https://arxiv.org/pdf/1807.09433). In *Proceedings of AAAI 2019*.
