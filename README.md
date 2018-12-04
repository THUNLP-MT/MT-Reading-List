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
* Libin Shen, Jinxi Xu, and Ralph Weischedel. 2009. [A New String-to-Dependency Machine Translation Algorithm with a Target Dependency Language Model](http://aclweb.org/anthology/P08-1066). In *Proceedings of ACL 2008*.

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
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. [Attention is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). In *Proceedings of NIPS 2017*.
* Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Mike Schuster, Noam Shazeer, Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Zhifeng Chen, Yonghui Wu, and Macduff Hughes. 2018. [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation](http://aclweb.org/anthology/P18-1008). In *Proceedings of ACL 2018*.

<h3 id="attention_mechanism">Attention Mechanism</h3>

* Biao Zhang, Deyi Xiong, and Jinsong Su. 2018. [Accelerating Neural Transformer via an Average Attention Network](http://aclweb.org/anthology/P18-1166). In *Proceedings of ACL 2018*.
* Tobias Domhan. 2018. [How Much Attention Do You Need? A Granular Analysis of Neural Machine Translation Architectures](http://aclweb.org/anthology/P18-1167). In *Proceedings of ACL 2018*.
* Shaohui Kuang, Junhui Li, António Branco, Weihua Luo, and Deyi Xiong. 2018. [Attention Focusing for Neural Machine Translation by Bridging Source and Target Embeddings](http://aclweb.org/anthology/P18-1164). In *Proceedings of ACL 2018*.

<h3 id="open_vocabulary">Open Vocabulary</h3>

* Thang Luong, Ilya Sutskever, Quoc Le, Oriol Vinyals, and Wojciech Zaremba. 2015. [Addressing the Rare Word Problem in Neural Machine Translation](http://aclweb.org/anthology/P15-1002). In *Proceedings of ACL 2015*.
* Sébastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. 2015. [On Using Very Large Target Vocabulary for Neural Machine Translation](http://www.aclweb.org/anthology/P15-1001). In *Proceedings of ACL 2015*.
* Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf). In *Proceedings of ACL 2016*.
* Taku Kudo. 2018. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](http://aclweb.org/anthology/P18-1007). In *Proceedings of ACL 2018*.

<h3 id="training">Training</h3>

* Shiqi Shen, Yong Cheng, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. [Minimum Risk Training for Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_mrt.pdf). In *Proceedings of ACL 2016*.
* Sam Wiseman and Alexander M. Rush. 2016. [Sequence-to-Sequence Learning as Beam-Search Optimization](http://aclweb.org/anthology/D16-1137). In *Proceedings of EMNLP 2006*.

<h3 id="decoding">Decoding</h3>

* Philip Schulz, Wilker Aziz, and Trevor Cohn. 2018. [A Stochastic Decoder for Neural Machine Translation](http://aclweb.org/anthology/P18-1115). In *Proceedings of ACL 2018*.

<h3 id="low_resource_language_translation">Low-resource Language Translation</h3>

* Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. [Improving Neural Machine Translation Models with Monolingual Data](https://arxiv.org/pdf/1511.06709). In *Proceedings of ACL 2016*.
* Yong Cheng, Wei Xu, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. [Semi-Supervised Learning for Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_semi.pdf). In *Proceedings of ACL 2016*.
* Barret Zoph, Deniz Yuret, Jonathan May, and Kevin Knight. 2016. [Transfer Learning for Low-Resource Neural Machine Translation](https://www.isi.edu/natural-language/mt/emnlp16-transfer.pdf). In *Proceedings of EMNLP 2016*.
* Yun Chen, Yang Liu, Yong Cheng and Victor O.K. Li. 2017. [A Teacher-Student Framework for Zero-resource Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_cy.pdf). In *Proceedings of ACL 2017*.
* Hao Zheng, Yong Cheng, and Yang Liu. 2017. [Maximum Expected Likelihood Estimation for Zero-resource Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/ijcai2017_zh.pdf). In *Proceedings of IJCAI 2017*.
* Yong Cheng, Qian Yang, Yang Liu, Maosong Sun, and Wei Xu. 2017. [Joint Training for Pivot-based Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/ijcai2017_cy.pdf). In *Proceedings of IJCAI 2017*.
* Zhen Yang, Wei Chen, Feng Wang, and Bo Xu. 2018. [Unsupervised Neural Machine Translation with Weight Sharing](http://aclweb.org/anthology/P18-1005). In *Proceedings of ACL 2018*.
* Shuo Ren, Wenhu Chen, Shujie Liu, Mu Li, Ming Zhou, and Shuai Ma. 2018. [Triangular Architecture for Rare Language Translation](http://aclweb.org/anthology/P18-1006). In *Proceedings of ACL 2018*.

<h3 id="multiple_language_translation">Multiple Language Translation</h3>

* Daxiang Dong, Hua Wu, Wei He, Dianhai Yu, and Haifeng Wang. 2015. [Multi-Task Learning for Multiple Language Translation](http://aclweb.org/anthology/P15-1166). In *Proceedings of ACL 2015*.
* Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda Viégas, Martin Wattenberg, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2017. [Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://arxiv.org/pdf/1611.04558). *Transactions of the Association for Computational Linguistics*.

<h3 id="prior_knowledge_integration">Prior Knowledge Integration</h3>

* Zhaopeng Tu, Zhengdong Lu, Yang Liu, Xiaohua Liu, and Hang Li. 2016. [Modeling Coverage for Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_coverage.pdf). In *Proceedings of ACL 2016*.
Yong Cheng, Shiqi Shen, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. [Agreement-based Joint Training for Bidirectional Attention-based Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/ijcai16_agree.pdf). In *Proceedings of IJCAI 2016*.
* Jiacheng Zhang, Yang Liu, Huanbo Luan, Jingfang Xu and Maosong Sun. 2017. [Prior Knowledge Integration for Neural Machine Translation using Posterior Regularization](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_zjc.pdf). In *Proceedings of ACL 2017*.
* Chris Hokamp and Qun Liu. 2017. [Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search](http://aclweb.org/anthology/P17-1141). In *Proceedings of ACL 2017*.
* Chunpeng Ma, Akihiro Tamura, Masao Utiyama, Tiejun Zhao, and Eiichiro Sumita. 2018. [Forest-Based Neural Machine Translation](http://aclweb.org/anthology/P18-1116). In *Proceedings of ACL 2018*.

<h3 id="document_level_translation">Document-level Translation</h3>

* Elena Voita, Pavel Serdyukov, Rico Sennrich, and Ivan Titov. 2018. [Context-Aware Neural Machine Translation Learns Anaphora Resolution](http://aclweb.org/anthology/P18-1117). In *Proceedings of ACL 2018*.
* Sameen Maruf and Gholamreza Haffari. 2018. [Document Context Neural Machine Translation with Memory Networks](http://aclweb.org/anthology/P18-1118). 2018. In *Proceedings of ACL 2018*.
* Jiacheng Zhang, Huanbo Luan, Maosong Sun, Feifei Zhai, Jingfang Xu, Min Zhang and Yang Liu. 2018. [Improving the Transformer Translation Model with Document-Level Context](http://aclweb.org/anthology/D18-1049). In *Proceedings of EMNLP 2018*.

<h3 id="robustness">Robustness</h3>

* Yong Cheng, Zhaopeng Tu, Fandong Meng, Junjie Zhai, and Yang Liu. 2018. [Towards Robust Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2018_cy.pdf). In *Proceedings of ACL 2018*.

<h3 id="visualization_and_interpretability">Visualization and Interpretability</h3>

* Yanzhuo Ding, Yang Liu, Huanbo Luan and Maosong Sun. 2017. [Visualizing and Understanding Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_dyz.pdf). In *Proceedings of ACL 2017*.

<h3 id="efficiency">Efficiency</h3>

* Abigail See, Minh-Thang Luong, and Christopher D. Manning. 2016. [Compression of Neural Machine Translation Models via Pruning](http://aclweb.org/anthology/K16-1029). In *Proceedings of CoNLL 2016*.

<h3 id="multi_modality">Multi-modality</h3>

* Iacer Calixto, Qun Liu, and Nick Campbell. 2017. [Doubly-Attentive Decoder for Multi-modal Neural Machine Translation](http://aclweb.org/anthology/P17-1175). In *Proceedings of ACL 2017*.
