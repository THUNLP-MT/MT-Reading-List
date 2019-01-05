# Machine Translation Reading List
This is a machine translation reading list maintained by the Tsinghua Natural Language Processing Group. 

The past three decades have witnessed the rapid development of machine translation, especially for data-driven approaches such as statistical machine translation (SMT) and neural machine translation (NMT). Due to the dominance of NMT at the present time, priority is given to collecting important, up-to-date NMT papers. The list is still incomplete and the categorization might be inappropriate. Each paper is given its Google Scholar citation count, which will be updated monthly.

We will keep adding papers and improving the list. Any suggestions are welcome!

* [10 Must Reads](#10_must_reads)
* [Statistical Machine Translation](#statistical_machine_translation)
    * [Tutorials](#smt_tutorials)
    * [Word-based Models](#word_based_models)
    * [Phrase-based Models](#phrase_based_models)
    * [Syntax-based Models](#syntax_based_models)
    * [Discriminative Training](#discriminative_training)
    * [System Combination](#system_combination)
    * [Evaluation](#evaluation)
 * [Neural Machine Translation](#neural_machine_translation)
    * [Tutorials and Challenges](#nmt_tutorials) 
    * [Model Architecture](#model_architecture)
    * [Attention Mechanism](#attention_mechanism)
    * [Open Vocabulary and Character-based NMT](#open_vocabulary)
    * [Training Objectives and Frameworks](#training)
    * [Decoding](#decoding)
    * [Low-resource Language Translation](#low_resource_language_translation)
        * [Semi-supervised Methods](#semi_supervised)
        * [Unsupervised Methods](#unsupervised)
        * [Pivot-based Methods](#pivot_based)
        * [Data Augmentation Methods](#data_augmentation)
        * [Data Selection Methods](#data_selection)
        * [Transfer Learning & Multi-Task Learning Methods](#transfer_learning)
        * [Meta Learning Methods](#meta_learning)
    * [Multilingual Language Translation](#multilingual_language_translation)
    * [Prior Knowledge Integration](#prior_knowledge_integration)
        * [Word/Phrase Constraints](#word_phrase_constraints)
        * [Syntactic/Semantic Constraints](#syntactic_semantic_constraints)
        * [Coverage Constraints](#coverage_constraints)
    * [Document-level Translation](#document_level_translation)
    * [Robustness](#robustness)
    * [Visualization and Interpretability](#visualization_and_interpretability)
    * [Linguistic Interpretation](#linguistic_interpretation)
    * [Fairness and Diversity](#fairness_and_diversity)
    * [Efficiency](#efficiency)
    * [Speech Translation and Simultaneous Translation](#speech_translation_and_simultaneous_translation)
    * [Multi-modality](#multi_modality)
    * [Pre-training](#pre_training)
    * [Domain Adaptation](#domain_adaptation)
    * [Quality Estimation](#quality_estimation)
    * [Automatic Post-Editing](#ape)
    * [Word Translation and Bilingual Lexicon Induction](#word_translation)
    * [Poetry Translation](#poetry_translation)

<h2 id="10_must_reads">10 Must Reads</h2> 

* Peter E. Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, and Robert L. Mercer. 1993. [The Mathematics of Statistical Machine Translation: Parameter Estimation](http://aclweb.org/anthology/J93-2003). *Computational Linguistics*. ([Citation](https://scholar.google.com/scholar?cites=2259057253133260714&as_sdt=2005&sciodt=0,5&hl=en): 4,965)
* Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. [BLEU: a Method for Automatic Evaluation of Machine Translation](http://aclweb.org/anthology/P02-1040). In *Proceedings of ACL 2002*. ([Citation](https://scholar.google.com/scholar?cites=9019091454858686906&as_sdt=2005&sciodt=0,5&hl=en): 8,507)
* Philipp Koehn, Franz J. Och, and Daniel Marcu. 2003. [Statistical Phrase-Based Translation](http://aclweb.org/anthology/N03-1017). In *Proceedings of NAACL 2003*. ([Citation](https://scholar.google.com/scholar?cites=11796378766060939113&as_sdt=2005&sciodt=0,5&hl=en): 3,514)
* Franz Josef Och. 2003. [Minimum Error Rate Training in Statistical Machine Translation](http://aclweb.org/anthology/P03-1021). In *Proceedings of ACL 2003*. ([Citation](https://scholar.google.com/scholar?cites=15358949031331886708&as_sdt=2005&sciodt=0,5&hl=en): 2,982)
* David Chiang. 2007. [Hierarchical Phrase-Based Translation](http://aclweb.org/anthology/J07-2003). *Computational Linguistics*. ([Citation](https://scholar.google.com.hk/scholar?cites=17074501474509484516&as_sdt=2005&sciodt=0,5&hl=en): 1,192)
* Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. [Sequence to Sequence Learning
with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf). In *Proceedings of NIPS 2014*. ([Citation](https://scholar.google.com/scholar?cites=13133880703797056141&as_sdt=2005&sciodt=0,5&hl=en): 5,428)
* Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf). In *Proceedings of ICLR 2015*. ([Citation](https://scholar.google.com/scholar?cites=9430221802571417838&as_sdt=2005&sciodt=0,5&hl=en): 5,572)
* Diederik P. Kingma, Jimmy Ba. 2015. [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980). In *Proceedings of ICLR 2015*. ([Citation](https://scholar.google.com/scholar?cites=16194105527543080940&as_sdt=2005&sciodt=0,5&hl=en): 16,572)
* Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf). In *Proceedings of ACL 2016*. ([Citation](https://scholar.google.com/scholar?cites=1307964014330144942&as_sdt=2005&sciodt=0,5&hl=en): 789)
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. [Attention is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). In *Proceedings of NIPS 2017*. ([Citation](https://scholar.google.com/scholar?cites=2960712678066186980&as_sdt=2005&sciodt=0,5&hl=en): 1,047)

<h2 id="statistical_machine_translation">Statistical Machine Translation</h2>

<h3 id="smt_tutorials">Tutorials</h3>

* Philipp Koehn. 2006. [Statistical Machine Translation: the Basic, the Novel, and the Speculative](http://homepages.inf.ed.ac.uk/pkoehn/publications/tutorial2006.pdf). *EACL 2006 Tutorial*. ([Citation](https://scholar.google.com.hk/scholar?cites=226053141145183075&as_sdt=2005&sciodt=0,5&hl=en): 10)
* Adam Lopez. 2008. [Statistical Machine Translation](http://delivery.acm.org/10.1145/1390000/1380586/a8-lopez.pdf?ip=101.5.129.50&id=1380586&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E587F3204F5B62A59%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1546058891_981e84a24804f2dbc0549b9892a2ea1d). *ACM Computing Surveys*. ([Citation](https://scholar.google.com.hk/scholar?cites=13327711981648149476&as_sdt=2005&sciodt=0,5&hl=en): 373)


<h3 id="word_based_models">Word-based Models</h3>

* Peter E. Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, and Robert L. Mercer. 1993. [The Mathematics of Statistical Machine Translation: Parameter Estimation](http://aclweb.org/anthology/J93-2003). *Computational Linguistics*. ([Citation](https://scholar.google.com/scholar?cites=2259057253133260714&as_sdt=2005&sciodt=0,5&hl=en): 4,965)
* Stephan Vogel, Hermann Ney, and Christoph Tillmann. 1996. [HMM-Based Word Alignment in Statistical Translation](http://aclweb.org/anthology/C96-2141). In *Proceedings of COLING 1996*. ([Citation](https://scholar.google.com.hk/scholar?cites=6742027174667056165&as_sdt=2005&sciodt=0,5&hl=en): 940)
* Franz Josef Och and Hermann Ney. 2003. [A Systematic Comparison of Various Statistical Alignment Models](http://aclweb.org/anthology/J03-1002). *Computational Linguistics*. ([Citation](https://scholar.google.com.hk/scholar?cites=7906670690027479083&as_sdt=2005&sciodt=0,5&hl=en): 3,980)
* Percy Liang, Ben Taskar, and Dan Klein. 2006. [Alignment by Agreement](https://cs.stanford.edu/~pliang/papers/alignment-naacl2006.pdf). In *Proceedings of NAACL 2006*. ([Citation](https://scholar.google.com.hk/scholar?cites=10766838746666771394&as_sdt=2005&sciodt=0,5&hl=en): 452)
* Chris Dyer, Victor Chahuneau, and Noah A. Smith. 2013. [A Simple, Fast, and Effective Reparameterization of IBM Model 2](http://www.aclweb.org/anthology/N13-1073). In *Proceedings of NAACL 2013*. ([Citation](https://scholar.google.com.hk/scholar?cites=13560076980956479370&as_sdt=2005&sciodt=0,5&hl=en): 310)

<h3 id="phrase_based_models">Phrase-based Models</h3>

* Philipp Koehn, Franz J. Och, and Daniel Marcu. 2003. [Statistical Phrase-Based Translation](http://aclweb.org/anthology/N03-1017). In *Proceedings of NAACL 2003*. ([Citation](https://scholar.google.com.hk/scholar?cites=11796378766060939113&as_sdt=2005&sciodt=0,5&hl=en): 3,516)
* Michel Galley and Christopher D. Manning. 2008. [A Simple and Effective Hierarchical Phrase Reordering Model](https://nlp.stanford.edu/pubs/emnlp08-lexorder.pdf). In *Proceedings of EMNLP 2008*. ([Citation](https://scholar.google.com.hk/scholar?cites=14572547803642015856&as_sdt=2005&sciodt=0,5&hl=en): 275)

<h3 id="syntax_based_models">Syntax-based Models</h3>

* Dekai Wu. 1997. [Stochastic Inversion Transduction Grammars and Bilingual Parsing of Parallel Corpora](http://aclweb.org/anthology/J97-3002). *Computational Linguistics*. ([Citation](https://scholar.google.com.hk/scholar?cites=7926725626202301933&as_sdt=2005&sciodt=0,5&hl=en): 1,009)
* Michel Galley, Jonathan Graehl, Kevin Knight, Daniel Marcu, Steve DeNeefe, Wei Wang, and Ignacio Thayer. 2006. [Scalable Inference and Training of Context-Rich Syntactic Translation Models](http://aclweb.org/anthology/P06-1121). In *Proceedings of COLING/ACL 2006*. ([Citation](https://scholar.google.com.hk/scholar?cites=2650671041278094269&as_sdt=2005&sciodt=0,5&hl=en): 475)
* Yang Liu, Qun Liu, and Shouxun Lin. 2006. [Tree-to-String Alignment Template for Statistical Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/P06-1077.pdf). In *Proceedings of COLING/ACL 2006*. ([Citation](https://scholar.google.com.hk/scholar?cites=8683308453323663525&as_sdt=2005&sciodt=0,5&hl=en): 391)
* Deyi Xiong, Qun Liu, and Shouxun Lin. 2006. [Maximum Entropy Based Phrase Reordering Model for Statistical Machine Translation](https://aclanthology.info/pdf/P/P06/P06-1066.pdf). In *Proceedings of COLING/ACL 2006*. ([Citation](https://scholar.google.com.hk/scholar?cites=11896300896063367737&as_sdt=2005&sciodt=0,5&hl=en): 299)
* David Chiang. 2007. [Hierarchical Phrase-Based Translation](http://aclweb.org/anthology/J07-2003). *Computational Linguistics*. ([Citation](https://scholar.google.com.hk/scholar?cites=17074501474509484516&as_sdt=2005&sciodt=0,5&hl=en): 1,192)
* Liang Huang and David Chiang. 2007. [Forest Rescoring: Faster Decoding with Integrated Language Models](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.88.5058&rep=rep1&type=pdf). In *Proceedings of ACL 2007*. ([Citation](https://scholar.google.com.hk/scholar?cites=2826188279623417237&as_sdt=2005&sciodt=0,5&hl=en): 280)
* Haitao Mi, Liang Huang, and Qun Liu. 2008. [Forest-based Translation](http://aclweb.org/anthology/P08-1023). *In Proceedings of ACL 2008*. ([Citation](https://scholar.google.com.hk/scholar?cites=11263493281241243162&as_sdt=2005&sciodt=0,5&hl=en): 239)
* Min Zhang, Hongfei Jiang, Aiti Aw, Haizhou Li, Chew Lim Tan, and Sheng Li. 2008. [A Tree Sequence Alignment-based Tree-to-Tree Translation Model](http://www.aclweb.org/anthology/P08-1064). In *Proceedings of ACL 2008*. ([Citation](https://scholar.google.com.hk/scholar?cites=4828105603038412208&as_sdt=2005&sciodt=0,5&hl=en): 124)
* Libin Shen, Jinxi Xu, and Ralph Weischedel. 2008. [A New String-to-Dependency Machine Translation Algorithm with a Target Dependency Language Model](http://aclweb.org/anthology/P08-1066). In *Proceedings of ACL 2008*. ([Citation](https://scholar.google.com.hk/scholar?cites=15082517325172081801&as_sdt=2005&sciodt=0,5&hl=en): 278)
* Haitao Mi and Liang Huang. 2008. [Forest-based Translation Rule Extraction](http://aclweb.org/anthology/D08-1022). In *Proceedings of EMNLP 2008*. ([Citation](https://scholar.google.com.hk/scholar?cites=11263493281241243162&as_sdt=2005&sciodt=0,5&hl=en): 239)
* Yang Liu, Yajuan Lü, and Qun Liu. 2009. [Improving Tree-to-Tree Translation with Packed Forests](http://aclweb.org/anthology/P09-1063). In *Proceedings of ACL/IJNLP 2009*. ([Citation](https://scholar.google.com.hk/scholar?cites=3907324274083528908&as_sdt=2005&sciodt=0,5&hl=en): 93)
* David Chiang. 2010. [Learning to Translate with Source and Target Syntax](http://aclweb.org/anthology/P10-1146). In *Proceedings of ACL 2010*. ([Citation](https://scholar.google.com.hk/scholar?cites=18270412258769590027&as_sdt=2005&sciodt=0,5&hl=en): 118)

<h3 id="discriminative_training">Discriminative Training</h3>

* Franz Josef Och and Hermann Ney. 2002. [Discriminative Training and Maximum Entropy Models for Statistical Machine Translation](http://aclweb.org/anthology/P02-1038). In *Proceedings of ACL 2002*. ([Citation](https://scholar.google.com.hk/scholar?cites=2845378992177918439&as_sdt=2005&sciodt=0,5&hl=en): 1,258)
* Franz Josef Och. 2003. [Minimum Error Rate Training in Statistical Machine Translation](http://aclweb.org/anthology/P03-1021). In *Proceedings of ACL 2003*. ([Citation](https://scholar.google.com.hk/scholar?cites=15358949031331886708&as_sdt=2005&sciodt=0,5&hl=en): 2,984)
* Taro Watanabe, Jun Suzuki, Hajime Tsukada, and Hideki Isozaki. 2007. [Online Large-Margin Training for Statistical Machine Translation](http://aclweb.org/anthology/D07-1080). In *Proceedings of EMNLP-CoNLL 2007*. ([Citation](https://scholar.google.com.hk/scholar?cites=6690339336101573833&as_sdt=2005&sciodt=0,5&hl=en): 197)
* David Chiang, Kevin Knight, and Wei Wang. 2009. [11,001 New Features for Statistical Machine Translation](http://aclweb.org/anthology/N09-1025). In *Proceedings of NAACL 2009*. ([Citation](https://scholar.google.com.hk/scholar?cites=14062409519286340147&as_sdt=2005&sciodt=0,5&hl=en): 251)

<h3 id="system_combination">System Combination</h3>

* Antti-Veikko Rosti, Spyros Matsoukas, and Richard Schwartz. 2007. [Improved Word-Level System Combination for Machine Translation](http://aclweb.org/anthology/P07-1040). In *Proceedings of ACL 2007*. ([Citation](https://scholar.google.com.hk/scholar?cites=13310846375895519088&as_sdt=2005&sciodt=0,5&hl=en): 144)
* Xiaodong He, Mei Yang, Jianfeng Gao, Patrick Nguyen, and Robert Moore. 2008. [Indirect-HMM-based Hypothesis Alignment for Combining Outputs from Machine Translation Systems](http://aclweb.org/anthology/D08-1011). In *Proceedings of EMNLP 2008*. ([Citation](https://scholar.google.com.hk/scholar?cites=5843300493006970528&as_sdt=2005&sciodt=0,5&hl=en): 96)

<h3 id="evaluation">Evaluation</h3>

* Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. [BLEU: a Method for Automatic Evaluation of Machine Translation](http://aclweb.org/anthology/P02-1040). In *Proceedings of ACL 2002*. ([Citation](https://scholar.google.com.hk/scholar?cites=9019091454858686906&as_sdt=2005&sciodt=0,5&hl=en): 8,499)
* Philipp Koehn. 2004. [Statistical Significance Tests for Machine Translation Evaluation](http://www.aclweb.org/anthology/W04-3250). In *Proceedings of EMNLP 2004*. ([Citation](https://scholar.google.com.hk/scholar?cites=6141850486206753388&as_sdt=2005&sciodt=0,5&hl=en): 1,015)
* Satanjeev Banerjee and Alon Lavie. 2005. [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](http://aclweb.org/anthology/W05-0909). In *Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization*. ([Citation](https://scholar.google.com.hk/scholar?cites=11797833340491598355&as_sdt=2005&sciodt=0,5&hl=en): 1,355)
* Matthew Snover and Bonnie Dorr, Richard Schwartz, Linnea Micciulla, and John Makhoul. 2006. [A Study of Translation Edit Rate with Targeted Human Annotation](http://mt-archive.info/AMTA-2006-Snover.pdf). In *Proceedings of AMTA 2006*.   ([Citation](https://scholar.google.com.hk/scholar?cites=1809540661740640949&as_sdt=2005&sciodt=0,5&hl=en): 1,713) 
* Xin Wang, Wenhu Chen, Yuan-Fang Wang, and William Yang Wang. 2018. [No Metrics Are Perfect: Adversarial Reward Learning for Visual Storytelling](http://aclweb.org/anthology/P18-1083). In *Proceedings of ACL 2018*. ([Citation](https://scholar.google.com.hk/scholar?cites=1809540661740640949&as_sdt=2005&sciodt=0,5&hl=en): 10) 



<h2 id="neural_machine_translation">Neural Machine Translation</h2>

<h3 id="nmt_tutorials">Tutorials and Challenges</h3>

* Thang Luong, Kyunghyun Cho, and Christopher Manning. 2016. [Neural Machine Translation](https://nlp.stanford.edu/projects/nmt/Luong-Cho-Manning-NMT-ACL2016-v4.pdf). *ACL 2016 Tutorial*.
* Graham Neubig. 2017. [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/pdf/1703.01619.pdf). *arXiv:1703.01619*.
* Oriol Vinyals and Navdeep Jaitly. 2017. [Seq2Seq ICML Tutorial](https://docs.google.com/presentation/d/1quIMxEEPEf5EkRHc2USQaoJRC4QNX6_KomdZTBMBWjk/present?slide=id.p). *ICML 2017 Tutorial*.
* Philipp Koehn. 2017. [Neural Machine Translation](https://arxiv.org/abs/1709.07809). *arxiv:1709.07809*.
* Philipp Koehn and Rebecca Knowles. 2017. [Six Challenges for Neural Machine Translation](http://www.aclweb.org/anthology/W17-3204). In *Proceedings of the First Workshop on Neural Machine Translation*.

<h3 id="model_architecture">Model Architecture</h3>

* Nal Kalchbrenner and Phil Blunsom. 2013. [Recurrent Continuous Translation Models](http://aclweb.org/anthology/D13-1176). In *Proceedings of EMNLP 2013*.
* Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. [Sequence to Sequence Learning
with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf). In *Proceedings of NIPS 2014*.
* Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473). In *Proceedings of ICLR 2015*.
* Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2016. [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144). In *Proceedings of NIPS 2016*.
* Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. 2016. [Deep Recurrent Models with Fast-Forward Connections for Neural Machine Translation](http://aclweb.org/anthology/Q16-1027). *Transactions of the Association for Computational Linguistics*.
* Ekaterina Garmash, and Christof Monz. 2016. [Ensemble Learning for Multi-Source Neural Machine Translation](http://aclweb.org/anthology/C16-1133). In *Proceedings of COLING 2016*.
* Biao Zhang, Deyi Xiong, Jinsong Su, Hong Duan, and Min Zhang. 2016. [Variational Neural Machine Translation](http://aclweb.org/anthology/D16-1050). In *Proceedings of EMNLP 2016*.
* Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. 2017. [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf). In *Proceedings of ICML 2017*.
* Jonas Gehring, Michael Auli, David Grangier, and Yann Dauphin. 2017. [A Convolutional Encoder Model for Neural Machine Translation](http://aclweb.org/anthology/P17-1012). In *Proceedings of ACL 2017*.
* Mingxuan Wang, Zhengdong Lu, Jie Zhou, and Qun Liu. 2017. [Deep Neural Machine Translation with Linear Associative Unit](http://aclweb.org/anthology/P17-1013). In *Proceedings of ACL 2017*.
* Long Zhou, Wenpeng Hu, Jiajun Zhang, and Chengqing Zong. 2017. [Neural System Combination for Machine Translation](http://aclweb.org/anthology/P17-2060). In *Proceedings of ACL 2017*.
* Matthias Sperber, Graham Neubig, Jan Niehues, and Alex Waibel. 2017. [Neural Lattice-to-Sequence Models for Uncertain Inputs](http://aclweb.org/anthology/D17-1145). In *Proceedings of EMNLP 2017*. 
* Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc Le. 2017. [Massive Exploration of Neural Machine Translation Architectures](http://aclweb.org/anthology/D17-1151). In *Proceedings of EMNLP 2017*. 
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. [Attention is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). In *Proceedings of NIPS 2017*.
* Lukasz Kaiser, Aidan N. Gomez, and Francois Chollet. 2018. [Depthwise Separable Convolutions for Neural Machine Translation](https://openreview.net/pdf?id=S1jBcueAb). In *Proceedings of ICLR 2018*.
* Łukasz Kaiser, Aurko Roy, Ashish Vaswani, Niki Parmar, Samy Bengio, Jakob Uszkoreit, and Noam Shazeer. 2018. [Fast Decoding in Sequence Models Using Discrete Latent Variables](https://arxiv.org/pdf/1803.03382.pdf). In *Proceedings of ICML 2018*.
* Yanyao Shen, Xu Tan, Di He, Tao Qin, and Tie-Yan Liu. 2018. [Dense Information Flow for Neural Machine Translation](http://aclweb.org/anthology/N18-1117). In *Proceedings of NAACL 2018*.
* Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Mike Schuster, Noam Shazeer, Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Zhifeng Chen, Yonghui Wu, and Macduff Hughes. 2018. [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation](http://aclweb.org/anthology/P18-1008). In *Proceedings of ACL 2018*.
* Weiyue Wang, Derui Zhu, Tamer Alkhouli, Zixuan Gan, and Hermann Ney. 2018. [Neural Hidden Markov Model for Machine Translation](http://aclweb.org/anthology/P18-2060). In *Proceedings of ACL 2018*.
* Qiang Wang, Fuxue Li, Tong Xiao, Yanyang Li, Yinqiao Li, and Jingbo Zhu. 2018. [Multi-layer Representation Fusion for Neural Machine Translation](http://aclweb.org/anthology/C18-1255). In *Proceedings of COLING 2018*.
* Yachao Li, Junhui Li, and Min Zhang. 2018. [Adaptive Weighting for Neural Machine Translation](http://aclweb.org/anthology/C18-1257). In *Proceedings of COLING 2018*. 
* Zi-Yi Dou, Zhaopeng Tu, Xing Wang, Shuming Shi, and Tong Zhang. 2018. [Exploiting Deep Representations for Neural Machine Translation](http://aclweb.org/anthology/D18-1457). In *Proceedings of EMNLP 2018*.
* Biao Zhang, Deyi Xiong, Jinsong Su, Qian Lin, and Huiji Zhang. 2018. [Simplifying Neural Machine Translation with Addition-Subtraction Twin-Gated Recurrent Networks](http://aclweb.org/anthology/D18-1459). In *Proceedings of EMNLP 2018*.
* Gongbo Tang, Mathias Müller, Annette Rios, and Rico Sennrich. 2018. [Why Self-Attention? A Targeted Evaluation of Neural Machine Translation Architectures](http://aclweb.org/anthology/D18-1458). In *Proceedings of EMNLP 2018*.
* Ke Tran, Arianna Bisazza, and Christof Monz. 2018. [The Importance of Being Recurrent for Modeling Hierarchical Structure](http://aclweb.org/anthology/D18-1503). In *Proceedings of EMNLP 2018*.
* Parnia Bahar, Christopher Brix, and Hermann Ney. 2018. [Towards Two-Dimensional Sequence to Sequence Model in Neural Machine Translation](http://aclweb.org/anthology/D18-1335). In *Proceedings of EMNLP 2018*.
* Tianyu He, Xu Tan, Yingce Xia, Di He, Tao Qin, Zhibo Chen, and Tie-Yan Liu. 2018. [Layer-Wise Coordination between Encoder and Decoder for Neural Machine Translation](http://papers.nips.cc/paper/8019-layer-wise-coordination-between-encoder-and-decoder-for-neural-machine-translation.pdf). In *Proceedings of NeurIPS 2018*.
* Hany Hassan, Anthony Aue, Chang Chen, Vishal Chowdhary, Jonathan Clark, Christian Federmann, Xuedong Huang, Marcin Junczys-Dowmunt, William Lewis, Mu Li, Shujie Liu, Tie-Yan Liu, Renqian Luo, Arul Menezes, Tao Qin, Frank Seide, Xu Tan, Fei Tian, Lijun Wu, Shuangzhi Wu, Yingce Xia, Dongdong Zhang, Zhirui Zhang, and Ming Zhou. 2018. [Achieving Human Parity on Automatic Chinese to English News Translation](https://www.microsoft.com/en-us/research/uploads/prod/2018/03/final-achieving-human.pdf). Technical report. Microsoft AI & Research.
* Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Lukasz Kaiser. 2019. [Universal Transformers](https://openreview.net/pdf?id=HyzdRiR9Y7). In *Proceedings of ICLR 2019*.

<h3 id="attention_mechanism">Attention Mechanism</h3>

* Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473). In *Proceedings of ICLR 2015*.
* Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025). In *Proceedings of EMNLP 2015*.
* Haitao Mi, Zhiguo Wang, and Abe Ittycheriah. 2016. [Supervised Attentions for Neural Machine Translation](http://aclweb.org/anthology/D16-1249). In *Proceedings of EMNLP 2016*.
* Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. 2017. [A structured self-attentive sentence embedding](https://arxiv.org/abs/1703.03130). In *Proceedings of ICLR 2017*. 
* Tao Shen, Tianyi Zhou, Guodong Long, Jing Jiang, Shirui Pan, and Chengqi Zhang. 2018. [DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding](https://arxiv.org/pdf/1709.04696.pdf). In *Proceedings of AAAI 2018*.
* Tao Shen, Tianyi Zhou, Guodong Long, Jing Jiang, and Chengqi Zhang. 2018. [Bi-directional block self-attention for fast and memory-efficient sequence modeling](https://arxiv.org/abs/1804.00857). In *Proceedings of ICLR 2018*. 
* Tao Shen, Tianyi Zhou, Guodong Long, Jing Jiang, Sen Wang, Chengqi Zhang. 2018.  [Reinforced Self-Attention Network: a Hybrid of Hard and Soft Attention for Sequence Modeling](https://arxiv.org/abs/1801.10296). In *Proceedings of IJCAI 2018*.
* Peter Shaw, Jakob Uszkorei, and Ashish Vaswani. 2018. [Self-Attention with Relative Position Representations](http://aclweb.org/anthology/N18-2074). In *Proceedings of NAACL 2018*.
* Lesly Miculicich Werlen, Nikolaos Pappas, Dhananjay Ram, and Andrei Popescu-Belis. 2018. [Self-Attentive Residual Decoder for Neural Machine Translation](http://aclweb.org/anthology/N18-1124). In *Proceedings of NAACL 2018*.
* Xintong Li, Lemao Liu, Zhaopeng Tu, Shuming Shi, and Max Meng. 2018. [Target Foresight Based Attention for Neural Machine Translation](http://aclweb.org/anthology/N18-1125). In *Proceedings of NAACL 2018*.
* Biao Zhang, Deyi Xiong, and Jinsong Su. 2018. [Accelerating Neural Transformer via an Average Attention Network](http://aclweb.org/anthology/P18-1166). In *Proceedings of ACL 2018*.
* Tobias Domhan. 2018. [How Much Attention Do You Need? A Granular Analysis of Neural Machine Translation Architectures](http://aclweb.org/anthology/P18-1167). In *Proceedings of ACL 2018*.
* Shaohui Kuang, Junhui Li, António Branco, Weihua Luo, and Deyi Xiong. 2018. [Attention Focusing for Neural Machine Translation by Bridging Source and Target Embeddings](http://aclweb.org/anthology/P18-1164). In *Proceedings of ACL 2018*.
* Chaitanya Malaviya, Pedro Ferreira, and André F. T. Martins. 2018. [Sparse and Constrained Attention for Neural Machine Translation](http://aclweb.org/anthology/P18-2059). In *Proceedings of ACL 2018*.
* Jian Li, Zhaopeng Tu, Baosong Yang, Michael R. Lyu, and Tong Zhang. 2018. [Multi-Head Attention with Disagreement Regularization](http://aclweb.org/anthology/D18-1317). In *Proceedings of EMNLP 2018*.
* Wei Wu, Houfeng Wang, Tianyu Liu and Shuming Ma.  2018. [Phrase-level Self-Attention Networks for Universal Sentence Encoding](http://aclweb.org/anthology/D18-1408). In *Proceedings of EMNLP 2018*.
* Baosong Yang, Zhaopeng Tu, Derek F. Wong, Fandong Meng, Lidia S. Chao, and Tong Zhang. 2018. [Modeling Localness for Self-Attention Networks](https://arxiv.org/abs/1810.10182). In *Proceedings of EMNLP 2018*.
* Junyang Lin, Xu Sun, Xuancheng Ren, Muyu Li, and Qi Su. 2018. [Learning When to Concentrate or Divert Attention: Self-Adaptive Attention Temperature for Neural Machine Translation](http://aclweb.org/anthology/D18-1331). In *Proceedings of EMNLP 2018*.
* Ankur Bapna, Mia Chen, Orhan Firat, Yuan Cao, and Yonghui Wu. 2018. [Training Deeper Neural Machine Translation Models with Transparent Attention](http://aclweb.org/anthology/D18-1338). In *Proceedings of EMNLP 2018*.
* Maha Elbayad, Laurent Besacier, and Jakob Verbeek. 2018. [Pervasive Attention: {2D} Convolutional Neural Networks for Sequence-to-Sequence Prediction](http://aclweb.org/anthology/K18-1010). In *Proceedings of CoNLL 2018*.

<h3 id="open_vocabulary">Open Vocabulary and Character-based NMT</h3>

* Felix Hill, Kyunghyun Cho, Sebastien Jean, Coline Devin, and Yoshua Bengio. 2015. [Embedding Word Similarity with Neural Machine Translation](https://arxiv.org/pdf/1412.6448.pdf). In *Proceedings of ICLR 2015*.
* Thang Luong, Ilya Sutskever, Quoc Le, Oriol Vinyals, and Wojciech Zaremba. 2015. [Addressing the Rare Word Problem in Neural Machine Translation](http://aclweb.org/anthology/P15-1002). In *Proceedings of ACL 2015*.
* Sébastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. 2015. [On Using Very Large Target Vocabulary for Neural Machine Translation](http://www.aclweb.org/anthology/P15-1001). In *Proceedings of ACL 2015*.
* Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf). In *Proceedings of ACL 2016*.
* Minh-Thang Luong and Christopher D. Manning. 2016. [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](http://aclweb.org/anthology/P16-1100). In *Proceedings of ACL 2016*.
* Junyoung Chung, Kyunghyun Cho, and Yoshua Bengio. 2016. [A Character-level Decoder without Explicit Segmentation for Neural Machine Translation](http://aclweb.org/anthology/P16-1160). In *Proceedings of ACL 2016*.
* Jason Lee, Kyunghyun Cho, and Thomas Hofmann. 2017. [Fully Character-Level Neural Machine Translation without Explicit Segmentation](http://aclweb.org/anthology/Q17-1026). *Transactions of the Association for Computational Linguistics*.
* Yang Feng, Shiyue Zhang, Andi Zhang, Dong Wang, and Andrew Abel. 2017. [Memory-augmented Neural Machine Translation](http://aclweb.org/anthology/D17-1146). In *Proceedings of EMNLP 2017*.
* Baosong Yang, Derek F. Wong, Tong Xiao, Lidia S. Chao, and Jingbo Zhu. 2017. [Towards Bidirectional Hierarchical Representations for Attention-based Neural Machine Translation](http://aclweb.org/anthology/D17-1150). In *Proceedings of EMNLP 2017*. 
* Peyman Passban, Qun Liu, and Andy Way. 2018. [Improving Character-Based Decoding Using Target-Side Morphological Information for Neural Machine Translation](http://aclweb.org/anthology/N18-1006). In *Proceedings of NAACL 2018*.
* Huadong Chen, Shujian Huang, David Chiang, Xinyu Dai, and Jiajun Chen. 2018. [Combining Character and Word Information in Neural Machine Translation Using a Multi-Level Attention](http://aclweb.org/anthology/N18-1116). In *Proceedings of NAACL 2018*.
* Frederick Liu, Han Lu, and Graham Neubig. 2018. [Handling Homographs in Neural Machine Translation](http://aclweb.org/anthology/N18-1121). In *Proceedings of NAACL 2018*.
* Taku Kudo. 2018. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](http://aclweb.org/anthology/P18-1007). In *Proceedings of ACL 2018*.
* Makoto Morishita, Jun Suzuki, and Masaaki Nagata. 2018. [Improving Neural Machine Translation by Incorporating Hierarchical Subword Features](http://aclweb.org/anthology/C18-1052). In *Proceedings of COLING 2018*. 
* Yang Zhao, Jiajun Zhang, Zhongjun He, Chengqing Zong, and Hua Wu. 2018. [Addressing Troublesome Words in Neural Machine Translation](http://aclweb.org/anthology/D18-1036). In *Proceedings of EMNLP 2018*.
* Colin Cherry, George Foster, Ankur Bapna, Orhan Firat, and Wolfgang Macherey. 2018. [Revisiting Character-Based Neural Machine Translation with Capacity and Compression](http://aclweb.org/anthology/D18-1461). In *Proceedings of EMNLP 2018*.
* Rebecca Knowles and Philipp Koehn. 2018. [Context and Copying in Neural Machine Translation](http://aclweb.org/anthology/D18-1339). In *Proceedings of EMNLP 2018*.

<h3 id="training">Training Objectives and Frameworks</h3>

* Marc'Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. 2016. [Sequence Level Training with Recurrent Neural Networks](https://arxiv.org/pdf/1511.06732). In *Proceedings of ICLR 2016*.   
* Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. 2016. [Multi-task Sequence to Sequence Learning](https://arxiv.org/pdf/1511.06114). In *Proceedings of ICLR 2016*. 
* Shiqi Shen, Yong Cheng, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. [Minimum Risk Training for Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_mrt.pdf). In *Proceedings of ACL 2016*.   
* Sam Wiseman and Alexander M. Rush. 2016. [Sequence-to-Sequence Learning as Beam-Search Optimization](http://aclweb.org/anthology/D16-1137). In *Proceedings of EMNLP 2016*.   
* Di He, Yingce Xia, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, Wei-Ying Ma. 2016. [Dual Learning for Machine Translation](https://papers.nips.cc/paper/6469-dual-learning-for-machine-translation.pdf). In *Proceedings of NIPS 2016*.
* Dzmitry Bahdanau, Philemon Brakel, Kelvin Xu, Anirudh Goyal, Ryan Lowe, Joelle Pineau, Aaron Courville, and Yoshua Bengio. 2017. [An Actor-Critic Algorithm for Sequence Prediction](https://arxiv.org/pdf/1607.07086). In *Proceedings of ICLR 2017*.   
* Khanh Nguyen, Hal Daumé III, and Jordan Boyd-Graber. 2017. [Reinforcement Learning for Bandit Neural Machine Translation with Simulated Human Feedback](http://aclweb.org/anthology/D17-1153). In *Proceedings of EMNLP 2017*.
* Nima Pourdamghani and Kevin Knight. 2017. [Deciphering Related Languages](http://aclweb.org/anthology/D17-1266). In *Proceedings of EMNLP 2017*. 
* Di He, Hanqing Lu, Yingce Xia, Tao Qin, Liwei Wang, and Tieyan Liu. 2017. [Decoding with Value Networks for Neural Machine Translation](http://papers.nips.cc/paper/6622-decoding-with-value-networks-for-neural-machine-translation.pdf). In *Proceedings of NIPS 2017*.
* Sergey Edunov, Myle Ott, Michael Auli, David Grangier, and Marc’Aurelio Ranzato. 2018. [Classical Structured Prediction Losses for Sequence to Sequence Learning](http://aclweb.org/anthology/N18-1033). In *Proceedings of NAACL 2018*.
* Zhen Yang, Wei Chen, Feng Wang, and Bo Xu. 2018. [Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets](http://aclweb.org/anthology/N18-1122). In *Proceedings of NAACL 2018*.    
* Pavel Petrushkov, Shahram Khadivi and Evgeny Matusov. 2018. [Learning from Chunk-based Feedback in Neural Machine Translation](http://aclweb.org/anthology/P18-2052). In *Proceedings of ACL 2018*.
* Lijun Wu, Fei Tian, Tao Qin, Jianhuang Lai, and Tie-Yan Liu. 2018. [A Study of Reinforcement Learning for Neural Machine Translation](http://aclweb.org/anthology/D18-1397). In *Proceedings of EMNLP 2018*.    
* Jiatao Gu, Yong Wang, Yun Chen, Kyunghyun Cho, and Victor O.K. Li. 2018. [Meta-Learning for Low-Resource Neural Machine Translation](http://aclweb.org/anthology/D18-1398). In *Proceedings of EMNLP 2018*.    
* Harshil Shah and David Barber. 2018. [Generative Neural Machine Translation](http://papers.nips.cc/paper/7409-generative-neural-machine-translation.pdf). In *Proceedings of NeurIPS 2018*.
* Lijun Wu, Fei Tian, Yingce Xia, Yang Fan, Tao Qin, Jianhuang Lai, and Tie-Yan Liu. 2018. [Learning to Teach with Dynamic Loss Functions](https://papers.nips.cc/paper/7882-learning-to-teach-with-dynamic-loss-functions.pdf). In *Proceedings of NeurIPS 2018*.
* Yiren Wang, Yingce Xia, Tianyu He, Fei Tian, Tao Qin, ChengXiang Zhai, and Tie-Yan Liu. 2019. [Multi-Agent Dual Learning](https://openreview.net/pdf?id=HyGhN2A5tm). In *Proceedings of ICLR 2019*.

<h3 id="decoding">Decoding</h3>

* Mingxuan Wang, Zhengdong Lu, Hang Li, and Qun Liu. 2016. [Memory-enhanced Decoder for Neural Machine Translation](http://aclweb.org/anthology/D16-1027). In *Proceedings of EMNLP 2016*.
* Shonosuke Ishiwatari, Jingtao Yao, Shujie Liu, Mu Li, Ming Zhou, Naoki Yoshinaga, Masaru Kitsuregawa, and Weijia Jia. 2017. [Chunk-based Decoder for Neural Machine Translation](http://aclweb.org/anthology/P17-1174). In *Proceedings of ACL 2017*.
* Hao Zhou, Zhaopeng Tu, Shujian Huang, Xiaohua Liu, Hang Li, and Jiajun Chen. 2017. [Chunk-Based Bi-Scale Decoder for Neural Machine Translation](http://aclweb.org/anthology/P17-2092). In *Proceedings of ACL 2017*.
* Zichao Yang, Zhiting Hu, Yuntian Deng, Chris Dyer, and Alex Smola. 2017. [Neural Machine Translation with Recurrent Attention Modeling](http://aclweb.org/anthology/E17-2061).  In *Proceedings of EACL 2017*.
* Cong Duy Vu Hoang, Gholamreza Haffari, and Trevor Cohn. 2017. [Towards Decoding as Continuous Optimisation in Neural Machine Translation](http://aclweb.org/anthology/D17-1014). In *Proceedings of EMNLP 2017*.
* Yin-Wen Chang and Michael Collins. 2017. [Source-Side Left-to-Right or Target-Side Left-to-Right? An Empirical Comparison of Two Phrase-Based Decoding Algorithms](http://aclweb.org/anthology/D17-1157). In *Proceedings of EMNLP 2017*.
* Jiatao Gu, Kyunghyun Cho, and Victor O.K. Li. 2017. [Trainable Greedy Decoding for Neural Machine Translation](http://aclweb.org/anthology/D17-1210). In *Proceedings of EMNLP 2017*. 
* Jiatao Gu, James Bradbury, Caiming Xiong, Victor O.K. Li, and Richard Socher. 2018. [Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281). In *Proceedings of ICLR 2018*.
* Xiangwen Zhang, Jinsong Su, Yue Qin, Yang Liu, Rongrong Ji, and Hongji Wang. 2018. [Asynchronous Bidirectional Decoding for Neural Machine Translation](https://arxiv.org/pdf/1801.05122). In *Proceedings of AAAI 2018*.
* Philip Schulz, Wilker Aziz, and Trevor Cohn. 2018. [A Stochastic Decoder for Neural Machine Translation](http://aclweb.org/anthology/P18-1115). In *Proceedings of ACL 2018*.
* Raphael Shu and Hideki Nakayama. 2018. [Improving Beam Search by Removing Monotonic Constraint for Neural Machine Translation](http://aclweb.org/anthology/P18-2054). In *Proceedings of ACL 2018*.
* Junyang Lin, Xu Sun, Xuancheng Ren, Shuming Ma, Jinsong Su, and Qi Su. 2018. [Deconvolution-Based Global Decoding for Neural Machine Translation](http://aclweb.org/anthology/C18-1276). In *Proceedings of COLING 2018*.
* Chunqi Wang, Ji Zhang, and Haiqing Chen. 2018. [Semi-Autoregressive Neural Machine Translation](http://aclweb.org/anthology/D18-1044). In *Proceedings of EMNLP 2018*.
* Xinwei Geng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2018. [Adaptive Multi-pass Decoder for Neural Machine Translation](http://aclweb.org/anthology/D18-1048). In *Proceedings of EMNLP 2018*.
* Wen Zhang, Liang Huang, Yang Feng, Lei Shen, and Qun Liu. 2018. [Speeding Up Neural Machine Translation Decoding by Cube Pruning](http://aclweb.org/anthology/D18-1460). In *Proceedings of EMNLP 2018*.
* Xinyi Wang, Hieu Pham, Pengcheng Yin, and Graham Neubig. 2018. [A Tree-based Decoder for Neural Machine Translation](http://aclweb.org/anthology/D18-1509). In *Proceedings of EMNLP 2018*.
* Chenze Shao, Xilin Chen, and Yang Feng. 2018. [Greedy Search with Probabilistic N-gram Matching for Neural Machine Translation](http://aclweb.org/anthology/D18-1510). In *Proceedings of EMNLP 2018*.
* Zhisong Zhang, Rui Wang, Masao Utiyama, Eiichiro Sumita, and Hai Zhao. 2018. [Exploring Recombination for Efficient Decoding of Neural Machine Translation](http://aclweb.org/anthology/D18-1511). In *Proceedings of EMNLP 2018*.
* Jetic Gū, Hassan S. Shavarani, and Anoop Sarkar. 2018. [Top-down Tree Structured Decoding with Syntactic Connections for Neural Machine Translation and Parsing](http://aclweb.org/anthology/D18-1037). In *Proceedings of EMNLP 2018*.
* Yilin Yang, Liang Huang, and Mingbo Ma. 2018. [Breaking the Beam Search Curse: A Study of (Re-)Scoring Methods and Stopping Criteria for Neural Machine Translation](http://aclweb.org/anthology/D18-1342). In *Proceedings of EMNLP 2018*.
* Yun Chen, Victor O.K. Li, Kyunghyun Cho, and Samuel R. Bowman. 2018. [A Stable and Effective Learning Strategy for Trainable Greedy Decoding](http://aclweb.org/anthology/D18-1035). In *Proceedings of EMNLP 2018*.

<h3 id="low_resource_language_translation">Low-resource Language Translation</h3>

<h4 id="semi_supervised">Semi-supervised Methods</h4>

* Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. [Improving Neural Machine Translation Models with Monolingual Data](https://arxiv.org/pdf/1511.06709). In *Proceedings of ACL 2016*.
* Yong Cheng, Wei Xu, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. [Semi-Supervised Learning for Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_semi.pdf). In *Proceedings of ACL 2016*.
* Tobias Domhan and Felix Hieber. 2017. [Using Target-side Monolingual Data for Neural Machine Translation through Multi-task Learning](http://aclweb.org/anthology/D17-1158). In *Proceedings of EMNLP 2017*.
* Anna Currey, Antonio Valerio Miceli Barone, and Kenneth Heafield. 2017. [Copied Monolingual Data Improves Low-Resource Neural Machine Translation](http://aclweb.org/anthology/W17-4715). In *Proceedings of the Second Conference on Machine Translation*. 

<h4 id="unsupervised">Unsupervised Methods</h4>

* Nima Pourdamghani and Kevin Knight. 2017. [Deciphering Related Languages](http://aclweb.org/anthology/D17-1266). In *Proceedings of EMNLP 2017*.
* Mikel Artetxe, Gorka Labaka, Eneko Agirre, and Kyunghyun Cho. 2018. [Unsupervised Neural Machine Translation](https://openreview.net/pdf?id=Sy2ogebAW). In *Proceedings of ICLR 2018*.
* Guillaume Lample, Alexis Conneau, Ludovic Denoyer, and Marc'Aurelio Ranzato. 2018. [Unsupervised Machine Translation Using Monolingual Corpora Only](https://openreview.net/pdf?id=rkYTTf-AZ). In *Proceedings of ICLR 2018*.
* Zhen Yang, Wei Chen, Feng Wang, and Bo Xu. 2018. [Unsupervised Neural Machine Translation with Weight Sharing](http://aclweb.org/anthology/P18-1005). In *Proceedings of ACL 2018*.
* Guillaume Lample, Myle Ott, Alexis Conneau, Ludovic Denoyer, and Marc'Aurelio Ranzato. 2018. [Phrase-Based & Neural Unsupervised Machine Translation](http://aclweb.org/anthology/D18-1549). In *Proceedings of EMNLP 2018*.
* Iftekhar Naim, Parker Riley, and Daniel Gildea. 2018. [Feature-Based Decipherment for Machine Translation](http://aclweb.org/anthology/J18-3006). *Computational Linguistics*.

<h4 id="pivot_based">Pivot-based Methods</h4>

* Orhan Firat, Baskaran Sankaran, Yaser Al-Onaizan, Fatos T. Yarman Vural, and Kyunghyun Cho. 2016. [Zero-Resource Translation with Multi-Lingual Neural Machine Translation](http://aclweb.org/anthology/D16-1026). In *Proceedings of EMNLP 2016*.
* Hao Zheng, Yong Cheng, and Yang Liu. 2017. [Maximum Expected Likelihood Estimation for Zero-resource Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/ijcai2017_zh.pdf). In *Proceedings of IJCAI 2017*.   
* Yun Chen, Yang Liu, Yong Cheng and Victor O.K. Li. 2017. [A Teacher-Student Framework for Zero-resource Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_cy.pdf). In *Proceedings of ACL 2017*.
* Yong Cheng, Qian Yang, Yang Liu, Maosong Sun, and Wei Xu. 2017. [Joint Training for Pivot-based Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/ijcai2017_cy.pdf). In *Proceedings of IJCAI 2017*.   
* Yun Chen, Yang Liu, and Victor O. K. Li. 2018. [Zero-Resource Neural Machine Translation with Multi-Agent Communication Game](https://arxiv.org/pdf/1802.03116). In *Proceedings of AAAI 2018*.
* Shuo Ren, Wenhu Chen, Shujie Liu, Mu Li, Ming Zhou, and Shuai Ma. 2018. [Triangular Architecture for Rare Language Translation](http://aclweb.org/anthology/P18-1006). In *Proceedings of ACL 2018*.

<h4 id="data_augmentation">Data Augmentation Methods</h4>

* Marzieh Fadaee, Arianna Bisazza, and Christof Monz. 2017. [Data Augmentation for Low-Resource Neural Machine Translation](http://aclweb.org/anthology/P17-2090). In *Proceedings of ACL 2017*.
* Marzieh Fadaee and Christof Monz. 2018. [Back-Translation Sampling by Targeting Difficult Words in Neural Machine Translation](http://aclweb.org/anthology/D18-1040). In *Proceedings of EMNLP 2018*.
* Sergey Edunov, Myle Ott, Michael Auli, and David Grangier. 2018. [Understanding Back-Translation at Scale](http://aclweb.org/anthology/D18-1045). In *Proceedings of EMNLP 2018*.
* Xinyi Wang, Hieu Pham, Zihang Dai, and Graham Neubig. 2018. [SwitchOut: an Efficient Data Augmentation Algorithm for Neural Machine Translation](http://aclweb.org/anthology/D18-1100). In *Proceedings of EMNLP 2018*.   

<h4 id="data_selection">Data Selection Methods</h4>

* Marlies van der Wees, Arianna Bisazza and Christof Monz. 2017. [Dynamic Data Selection for Neural Machine Translation](http://aclweb.org/anthology/D17-1147). In *Proceedings of EMNLP 2017*.
* Holger Schwenk. 2018. [Filtering and Mining Parallel Data in a Joint Multilingual Space](http://aclweb.org/anthology/P18-2037). In *Proceedings of ACL 2018*.

<h4 id="transfer_learning">Transfer Learning & Multi-Task Learning Methods</h4>

* Barret Zoph, Deniz Yuret, Jonathan May, and Kevin Knight. 2016. [Transfer Learning for Low-Resource Neural Machine Translation](https://www.isi.edu/natural-language/mt/emnlp16-transfer.pdf). In *Proceedings of EMNLP 2016*.
* Jiatao Gu, Hany Hassan, Jacob Devlin, and Victor O.K. Li. 2018. [Universal Neural Machine Translation for Extremely Low Resource Languages](http://aclweb.org/anthology/N18-1032). In *Proceedings of NAACL 2018*.
* Poorya Zaremoodi and Gholamreza Haffari. 2018. [Neural Machine Translation for Bilingually Scarce Scenarios: a Deep Multi-Task Learning Approach](http://aclweb.org/anthology/N18-1123). In *Proceedings of NAACL 2018*.
* Poorya Zaremoodi, Wray Buntine, and Gholamreza Haffari. 2018. [Adaptive Knowledge Sharing in Multi-Task Learning: Improving Low-Resource Neural Machine Translation](http://aclweb.org/anthology/P18-2104). In *Proceedings of ACL 2018*.
* Tom Kocmi and Ondřej Bojar. 2018. [Trivial Transfer Learning for Low-Resource Neural Machine Translation](http://aclweb.org/anthology/W18-6325). In *Proceedings of the Third Conference on Machine Translation: Research Papers*.

<h4 id="meta_learning">Meta Learning Methods</h4>

* Jiatao Gu, Yong Wang, Yun Chen, Kyunghyun Cho, and Victor O.K. Li. 2018. [Meta-Learning for Low-Resource Neural Machine Translation](http://aclweb.org/anthology/D18-1398). In *Proceedings of EMNLP 2018*.    


<h3 id="multilingual_language_translation">Multilingual Language Translation</h3>

* Daxiang Dong, Hua Wu, Wei He, Dianhai Yu, and Haifeng Wang. 2015. [Multi-Task Learning for Multiple Language Translation](http://aclweb.org/anthology/P15-1166). In *Proceedings of ACL 2015*.
* Orhan Firat, Kyunghyun Cho and Yoshua Bengio. 2016. [Multi-way, multilingual neural machine translation with a Shared Attention Mechanism](https://arxiv.org/pdf/1601.01073.pdf). In *Proceedings of NAACL 2016*.
* Barret Zoph and Kevin Knight. 2016. [Multi-Source Neural Translation](https://arxiv.org/pdf/1601.00710.pdf). In *Proceedings of NAACL 2016*.
* Orhan Firat, Baskaran SanKaran, Yaser Al-Onaizan, Fatos T.Yarman Vural, Kyunghyun Cho. 2016. [Zero-Resource Translation with Multi-Lingual Neural Machine Translation](https://arxiv.org/pdf/1606.04164.pdf). In *Proceedings of EMNLP 2016*.
* Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda Viégas, Martin Wattenberg, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2017. [Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://arxiv.org/pdf/1611.04558). *Transactions of the Association for Computational Linguistics*.
* Surafel Melaku Lakew, Mauro Cettolo, and Marcello Federico. 2018. [A Comparison of Transformer and Recurrent Neural Networks on Multilingual Neural Machine Translation](http://aclweb.org/anthology/C18-1054). In *Proceedings of COLING 2018*. 
* Graeme Blackwood, Miguel Ballesteros, and Todd Ward. 2018. [Multilingual Neural Machine Translation with Task-Specific Attention](http://aclweb.org/anthology/C18-1263). In *Proceedings of COLING 2018*.  
* Devendra Singh Sachan and Graham Neubig. 2018. [Parameter Sharing Methods for Multilingual Self-Attentional Translation Models](http://aclweb.org/anthology/W18-6327). In *Proceedings of the Third Conference on Machine Translation: Research Papers*.
* Emmanouil Antonios Platanios, Mrinmaya Sachan, Graham Neubig, and Tom Mitchell. 2018. [Contextual Parameter Generation for Universal Neural Machine Translation](http://aclweb.org/anthology/D18-1039). In *Proceedings of EMNLP 2018*.
* Yining Wang, Jiajun Zhang, Feifei Zhai, Jingfang Xu, and Chengqing Zong. 2018. [Three Strategies to Improve One-to-Many Multilingual Translation](http://aclweb.org/anthology/D18-1326). In *Proceedings of EMNLP 2018*.
* Xu Tan, Yi Ren, Di He, Tao Qin, Zhou Zhao, and Tie-Yan Liu. 2019. [Multilingual Neural Machine Translation with Knowledge Distillation](https://openreview.net/pdf?id=S1gUsoR9YX). In *Proceedings of ICLR 2019*.
* Xinyi Wang, Hieu Pham, Philip Arthur, and Graham Neubig. 2019. [Multilingual Neural Machine Translation With Soft Decoupled Encoding](https://openreview.net/pdf?id=Skeke3C5Fm). In *Proceedings of ICLR 2019*.

<h3 id="prior_knowledge_integration">Prior Knowledge Integration</h3>

<h4 id="word_phrase_constraints"> Word/Phrase Constraints </h4>

* Wei He, Zhongjun He, Hua Wu, and Haifeng Wang. 2016. [Improved nerual machine translation with SMT features](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12189/11577). In *Proceedings of AAAI 2016*.
* Haitao Mi, Zhiguo Wang, and Abe Ittycheriah. 2016. [Vocabulary Manipulation for Neural Machine Translation](http://anthology.aclweb.org/P16-2021). In *Proceedings of ACL 2016*.
* Philip Arthur, Graham Neubig, and Satoshi Nakamura. 2016. [Incorporating Discrete Translation Lexicons into Neural Machine Translation](http://aclweb.org/anthology/D16-1162). In *Proceedings of EMNLP 2016*.
* Jiacheng Zhang, Yang Liu, Huanbo Luan, Jingfang Xu and Maosong Sun. 2017. [Prior Knowledge Integration for Neural Machine Translation using Posterior Regularization](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_zjc.pdf). In *Proceedings of ACL 2017*.
* Chris Hokamp and Qun Liu. 2017. [Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search](http://aclweb.org/anthology/P17-1141). In *Proceedings of ACL 2017*.
* Zichao Yang, Zhiting Hu, Yuntian Deng, Chris Dyer, and Alex Smola. 2017. [Neural Machine Translation with Recurrent Attention Modeling](http://aclweb.org/anthology/E17-2061).  In *Proceedings of EACL 2017*.
* Rongxiang Weng, Shujian Huang, Zaixiang Zheng, Xinyu Dai, and Jiajun Chen. 2017. [Neural Machine Translation with Word Predictions](http://aclweb.org/anthology/D17-1013). In *Proceedings of EMNLP 2017*.
* Yang Feng, Shiyue Zhang, Andi Zhang, Dong Wang, and Andrew Abel. 2017. [Memory-augmented Neural Machine Translation](http://aclweb.org/anthology/D17-1146). In *Proceedings of EMNLP 2017*.
* Leonard Dahlmann, Evgeny Matusov, Pavel Petrushkov, and Shahram Khadivi. 2017. [Neural Machine Translation Leveraging Phrase-based Models in a Hybrid Search](http://aclweb.org/anthology/D17-1148). In *Proceedings of EMNLP 2017*.
* Xing Wang, Zhaopeng Tu, Deyi Xiong, and Min Zhang. 2017. [Translating Phrases in Neural Machine Translation](http://aclweb.org/anthology/D17-1149). In *Proceedings of EMNLP 2017*.
* Baosong Yang, Derek F. Wong, Tong Xiao, Lidia S. Chao, and Jingbo Zhu. 2017. [Towards Bidirectional Hierarchical Representations for Attention-based Neural Machine Translation](http://aclweb.org/anthology/D17-1150). In *Proceedings of EMNLP 2017*. 
* Po-Sen Huang, Chong Wang, Sitao Huang, Dengyong Zhou, and Li Deng. 2018. [Towards Neural Phrase-based Machine Translation](https://openreview.net/pdf?id=HktJec1RZ). In *Proceedings of ICLR 2018*.
* Toan Nguyen and David Chiang. 2018. [Improving Lexical Choice in Neural Machine Translation](http://aclweb.org/anthology/N18-1031). In *Proceedings of NAACL 2018*.
* Huadong Chen, Shujian Huang, David Chiang, Xinyu Dai, and Jiajun Chen. 2018. [Combining Character and Word Information in Neural Machine Translation Using a Multi-Level Attention](http://aclweb.org/anthology/N18-1116). In *Proceedings of NAACL 2018*.
* Matt Post and David Vilar. 2018. [Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation](http://aclweb.org/anthology/N18-1119). In *Proceedings of NAACL 2018*.
* Jingyi Zhang, Masao Utiyama, Eiichro Sumita, Graham Neubig, and Satoshi Nakamura. 2018. [Guiding Neural Machine Translation with Retrieved Translation Pieces](http://aclweb.org/anthology/N18-1120). In *Proceedings of NAACL 2018*.
* Eva Hasler, Adrià de Gispert, Gonzalo Iglesias, and Bill Byrne. 2018. [Neural Machine Translation Decoding with Terminology Constraints](http://aclweb.org/anthology/N18-2081). In *Proceedings of NAACL 2018*.
* Nima Pourdamghani, Marjan Ghazvininejad, and Kevin Knight. 2018. [Using Word Vectors to Improve Word Alignments for Low Resource Machine Translation](http://aclweb.org/anthology/N18-2083). In *Proceedings of NAACL 2018*.
* Shuming Ma, Xu SUN, Yizhong Wang, and Junyang Lin. 2018. [Bag-of-Words as Target for Neural Machine Translation](http://aclweb.org/anthology/P18-2053). In *Proceedings of ACL 2018*.
* Mingxuan Wang, Jun Xie, Zhixing Tan, Jinsong Su, Deyi Xiong, and Chao Bian. 2018. [Neural Machine Translation with Decoding-History Enhanced Attention](http://aclweb.org/anthology/C18-1124). In *Proceedings of COLING 2018*. 
* Arata Ugawa, Akihiro Tamura, Takashi Ninomiya, Hiroya Takamura, and Manabu Okumura. 2018. [Neural Machine Translation Incorporating Named Entity](http://aclweb.org/anthology/C18-1274). In *Proceedings of COLING 2018*. 
* Longyue Wang, Zhaopeng Tu, Andy Way, and Qun Liu. 2018. [Learning to Jointly Translate and Predict Dropped Pronouns with a Shared Reconstruction Mechanism](http://aclweb.org/anthology/D18-1333). In *Proceedings of EMNLP 2018*.
* Qian Cao and Deyi Xiong. 2018. [Encoding Gated Translation Memory into Neural Machine Translation](http://aclweb.org/anthology/D18-1340). In *Proceedings of EMNLP 2018*.
* Chengyue Gong, Di He, Xu Tan, Tao Qin, Liwei Wang, and Tie-Yan Liu. 2018. [FRAGE: Frequency-Agnostic Word Representation](https://arxiv.org/pdf/1809.06858). In *Proceedings of NeurIPS 2018*.

<h4 id="syntactic_semantic_constraints"> Syntactic/Semantic Constraints </h4>

* Trevor Cohn, Cong Duy Vu Hoang, Ekaterina Vymolova, Kaisheng Yao, Chris Dyer, and Gholamreza Haffari. 2016. [Incorporating Structural Alignment Biases into an Attentional Neural Translation Model](https://arxiv.org/pdf/1601.01085.pdf). In *Proceedings of NAACL 2016*.
* Yong Cheng, Shiqi Shen, Zhongjun He, Wei He, Hua Wu, Maosong Sun, and Yang Liu. 2016. [Agreement-based Joint Training for Bidirectional Attention-based Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/ijcai16_agree.pdf). In *Proceedings of IJCAI 2016*.
* Akiko Eriguchi, Kazuma Hashimoto, and Yoshimasa Tsuruoka. 2016. [Tree-to-Sequence Attentional Neural Machine Translation](http://aclweb.org/anthology/P16-1078). In *Proceedings of ACL 2016*.
* Junhui Li, Deyi Xiong, Zhaopeng Tu, Muhua Zhu, Min Zhang, and Guodong Zhou. 2017. [Modeling Source Syntax for Neural Machine Translation](http://aclweb.org/anthology/P17-1064). In *Proceedings of ACL 2017*.
* Shuangzhi Wu, Dongdong Zhang, Nan Yang, Mu Li, and Ming Zhou. 2017. [Sequence-to-Dependency Neural Machine Translation](http://aclweb.org/anthology/P17-1065). In *Proceedings of ACL 2017*.
* Jinchao Zhang, Mingxuan Wang, Qun Liu, and Jie Zhou. 2017. [Incorporating Word Reordering Knowledge into Attention-based Neural Machine Translation](http://aclweb.org/anthology/P17-1140). In *Proceedings of ACL 2017*.
* Huadong Chen, Shujian Huang, David Chiang, and Jiajun Chen. 2017. [Improved Neural Machine Translation with a Syntax-Aware Encoder and Decoder](http://aclweb.org/anthology/P17-1177). In *Proceedings of ACL 2017*.
* Akiko Eriguchi, Yoshimasa Tsuruoka, and Kyunghyun Cho. 2017. [Learning to Parse and Translate Improves Neural Machine Translation](http://aclweb.org/anthology/P17-2012). In *Proceedings of ACL 2017*.
* Roee Aharoni and Yoav Goldberg. 2017. [Towards String-To-Tree Neural Machine Translation](http://aclweb.org/anthology/P17-2021). In *Proceedings of ACL 2017*.
* Kazuma Hashimoto and Yoshimasa Tsuruoka. 2017. [Neural Machine Translation with Source-Side Latent Graph Parsing](http://aclweb.org/anthology/D17-1012). In *Proceedings of EMNLP 2017*.
* Joost Bastings, Ivan Titov, Wilker Aziz, Diego Marcheggiani, and Khalil Simaan. 2017. [Graph Convolutional Encoders for Syntax-aware Neural Machine Translation](http://aclweb.org/anthology/D17-1209). In *Proceedings of EMNLP 2017*.
* Kehai Chen, Rui Wang, Masao Utiyama, Lemao Liu, Akihiro Tamura, Eiichiro Sumita, and Tiejun Zhao. 2017. [Neural Machine Translation with Source Dependency Representation](http://aclweb.org/anthology/D17-1304). In *Proceedings of EMNLP 2017*. 
* Peyman Passban, Qun Liu, and Andy Way. 2018. [Improving Character-Based Decoding Using Target-Side Morphological Information for Neural Machine Translation](http://aclweb.org/anthology/N18-1006). In *Proceedings of NAACL 2018*.
* Diego Marcheggiani, Joost Bastings, and Ivan Titov. 2018. [Exploiting Semantics in Neural Machine Translation with Graph Convolutional Networks](http://aclweb.org/anthology/N18-2078). In *Proceedings of NAACL 2018*.
* Chunpeng Ma, Akihiro Tamura, Masao Utiyama, Tiejun Zhao, and Eiichiro Sumita. 2018. [Forest-Based Neural Machine Translation](http://aclweb.org/anthology/P18-1116). In *Proceedings of ACL 2018*.
* Shaohui Kuang, Junhui Li, António Branco, Weihua Luo, and Deyi Xiong. 2018. [Attention Focusing for Neural Machine Translation by Bridging Source and Target Embeddings](http://aclweb.org/anthology/P18-1164). In *Proceedings of ACL 2018*.
* Duygu Ataman and Marcello Federico. 2018. [Compositional Representation of Morphologically-Rich Input for Neural Machine Translation](http://aclweb.org/anthology/P18-2049). In *Proceedings of ACL 2018*.
* Danielle Saunders, Felix Stahlberg, Adrià de Gispert, and Bill Byrne. 2018. [Multi-representation ensembles and delayed SGD updates improve syntax-based NMT](http://aclweb.org/anthology/P18-2051). In *Proceedings of ACL 2018*.
* Wen Zhang, Jiawei Hu, Yang Feng, and Qun Liu. 2018. [Refining Source Representations with Relation Networks for Neural Machine Translation](http://aclweb.org/anthology/C18-1110). In *Proceedings of COLING 2018*. 
* Poorya Zaremoodi and Gholamreza Haffari. 2018. [Incorporating Syntactic Uncertainty in Neural Machine Translation with a Forest-to-Sequence Model](http://aclweb.org/anthology/C18-1120). In *Proceedings of COLING 2018*.
* Hao Zhang, Axel Ng, and Richard Sproat. 2018. [Fast and Accurate Reordering with ITG Transition RNN](http://aclweb.org/anthology/C18-1123). In *Proceedings of COLING 2018*.  
* Jetic Gū, Hassan S. Shavarani, and Anoop Sarkar. 2018. [Top-down Tree Structured Decoding with Syntactic Connections for Neural Machine Translation and Parsing](http://aclweb.org/anthology/D18-1037). In *Proceedings of EMNLP 2018*.
* Anna Currey and Kenneth Heafield. 2018. [Multi-Source Syntactic Neural Machine Translation](http://aclweb.org/anthology/D18-1327). In *Proceedings of EMNLP 2018*.
* Xinyi Wang, Hieu Pham, Pengcheng Yin, and Graham Neubig. 2018. [A Tree-based Decoder for Neural Machine Translation](http://aclweb.org/anthology/D18-1509). In *Proceedings of EMNLP 2018*.
* Eliyahu Kiperwasser and Miguel Ballesteros. 2018. [Scheduled Multi-Task Learning: From Syntax to Translation](http://aclweb.org/anthology/Q18-1017). *Transactions of the Association for Computational Linguistics*.

<h4 id="coverage_constraints">Coverage Constraints</h4>

* Zhaopeng Tu, Zhengdong Lu, Yang Liu, Xiaohua Liu, and Hang Li. 2016. [Modeling Coverage for Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_coverage.pdf). In *Proceedings of ACL 2016*.
* Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2016. [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144). In *Proceedings of NIPS 2016*.
* Haitao Mi, Baskaran Sankaran, Zhiguo Wang, and Abe Ittycheriah. 2016. [Coverage Embedding Models for Neural Machine Translation](http://aclweb.org/anthology/D16-1096). In *Proceedings of EMNLP 2016*.
* Yanyang Li, Tong Xiao, Yinqiao Li, Qiang Wang, Changming Xu, and Jingbo Zhu. 2018. [A Simple and Effective Approach to Coverage-Aware Neural Machine Translation](http://aclweb.org/anthology/P18-2047). In *Proceedings of ACL 2018*.
* Zaixiang Zheng, Hao Zhou, Shujian Huang, Lili Mou, Xinyu Dai, Jiajun Chen, and Zhaopeng Tu. 2018. [Modeling Past and Future for Neural Machine Translation](https://aclanthology.coli.uni-saarland.de/events/tacl-2018). *Transactions of the Association for Computational Linguistics*.
* Xiang Kong, Zhaopeng Tu, Shuming Shi, Eduard Hovy, and Tong Zhang. [Neural Machine Translation with Adequacy-Oriented Learning](https://arxiv.org/pdf/1811.08541.pdf). In *Proceedings of AAAI 2019*.

<h3 id="document_level_translation">Document-level Translation</h3>

* Longyue Wang, Zhaopeng Tu, Andy Way, and Qun Liu. 2017. [Exploiting Cross-Sentence Context for Neural Machine Translation](http://aclweb.org/anthology/D17-1301). In *Proceedings of EMNLP 2017*.
* Zhaopeng Tu, Yang Liu, Zhengdong Lu, Xiaohua Liu, and Hang Li. 2017. [Context Gates for Neural Machine Translation](http://aclweb.org/anthology/Q17-1007). *Transactions of the Association for Computational Linguistics*.
* Rachel Bawden, Rico Sennrich, Alexandra Birch, and Barry Haddow. 2018. [Evaluating Discourse Phenomena in Neural Machine Translation](http://aclweb.org/anthology/N18-1118). In *Proceedings of NAACL 2018*.
* Elena Voita, Pavel Serdyukov, Rico Sennrich, and Ivan Titov. 2018. [Context-Aware Neural Machine Translation Learns Anaphora Resolution](http://aclweb.org/anthology/P18-1117). In *Proceedings of ACL 2018*.
* Sameen Maruf and Gholamreza Haffari. 2018. [Document Context Neural Machine Translation with Memory Networks](http://aclweb.org/anthology/P18-1118). In *Proceedings of ACL 2018*.
* [Modeling Coherence for Neural Machine Translation with Dynamic and Topic Caches](http://aclweb.org/anthology/C18-1050). In *Proceedings of COLING 2018*.
* Shaohui Kuang and Deyi Xiong. 2018. [Fusing Recency into Neural Machine Translation with an Inter-Sentence Gate Model](https://arxiv.org/pdf/1806.04466.pdf). In *Proceedings of COLING 2018*. 
* Jiacheng Zhang, Huanbo Luan, Maosong Sun, Feifei Zhai, Jingfang Xu, Min Zhang and Yang Liu. 2018. [Improving the Transformer Translation Model with Document-Level Context](http://aclweb.org/anthology/D18-1049). In *Proceedings of EMNLP 2018*.
* Samuel Läubli, Rico Sennrich, and Martin Volk. 2018. [Has Machine Translation Achieved Human Parity? A Case for Document-level Evaluation](http://aclweb.org/anthology/D18-1512). In *Proceedings of EMNLP 2018*.
* Lesly Miculicich, Dhananjay Ram, Nikolaos Pappas, and James Henderson. 2018. [Document-Level Neural Machine Translation with Hierarchical Attention Networks](http://aclweb.org/anthology/D18-1325). In *Proceedings of EMNLP 2018*.
* Zhaopeng Tu, Yang Liu, Shumin Shi, and Tong Zhang. 2018. [Learning to Remember Translation History with a Continuous Cache](https://arxiv.org/pdf/1711.09367.pdf). *Transactions of the Association for Computational Linguistics*.

<h3 id="robustness">Robustness</h3>

* Yonatan Belinkov and Yonatan Bisk. 2018. [Synthetic and Natural Noise Both Break Neural Machine Translation](https://openreview.net/pdf?id=BJ8vJebC-). In *Proceedings of ICLR 2018*.
* Zhengli Zhao, Dheeru Dua, and Sameer Singh. 2018. [Generating Natural Adversarial Examples](https://openreview.net/pdf?id=H1BLjgZCb). In *Proceedings of ICLR 2018*.     
* Yong Cheng, Zhaopeng Tu, Fandong Meng, Junjie Zhai, and Yang Liu. 2018. [Towards Robust Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2018_cy.pdf). In *Proceedings of ACL 2018*.          
* Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2018. [Semantically Equivalent Adversarial Rules for Debugging NLP models](http://aclweb.org/anthology/P18-1079). In *Proceedings of ACL 2018*.          
* Javid Ebrahimi, Daniel Lowd, and Dejing Dou. 2018. [On Adversarial Examples for Character-Level Neural Machine Translation](http://aclweb.org/anthology/C18-1055). In *Proceedings of COLING 2018*.   
* Paul Michel and Graham Neubig. 2018. [MTNT: A Testbed for Machine Translation of Noisy Text](http://aclweb.org/anthology/D18-1050). In *Proceedings of EMNLP 2018*.

<h3 id="visualization_and_interpretability">Visualization and Interpretability</h3>

* Yanzhuo Ding, Yang Liu, Huanbo Luan and Maosong Sun. 2017. [Visualizing and Understanding Neural Machine Translation](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_dyz.pdf). In *Proceedings of ACL 2017*.
* Hendrik Strobelt, Sebastian Gehrmann, Michael Behrisch, Adam Perer, Hanspeter Pfister, and Alexander M. Rush. 2018. [Seq2Seq-Vis: A Visual Debugging Tool for Sequence-to-Sequence Models](https://arxiv.org/pdf/1804.09299.pdf). In *Proceedings of VAST 2018* and *Proceedings of EMNLP-BlackBox 2018*.
* Alessandro Raganato and Jorg Tiedemann. 2018. [An Analysis of Encoder Representations in Transformer-Based Machine Translation](http://aclweb.org/anthology/W18-5431). In *Proceedings of EMNLP-BlackBox 2018*.
* Felix Stahlberg, Danielle Saunders, and Bill Byrne. 2018. [An Operation Sequence Model for Explainable Neural Machine Translation](http://aclweb.org/anthology/W18-5420). In *Proceedings of EMNLP-BlackBox 2018*.
* Anthony Bau, Yonatan Belinkov, Hassan Sajjad, Nadir Durrani, Fahim Dalvi, and James Glass. 2019. [Identifying and Controlling Important Neurons in Neural Machine Translation](https://openreview.net/pdf?id=H1z-PsR5KX). In *Proceedings of ICLR 2019*.

<h3 id="linguistic_interpretation">Linguistic Interpretation</h3>

* Felix Hill, Kyunghyun Cho, Sebastien Jean, Coline Devin, and Yoshua Bengio. 2015. [Embedding Word Similarity with Neural Machine Translation](https://arxiv.org/pdf/1412.6448.pdf). In *Proceedings of ICLR 2015*.
* Yonatan Belinkov, Nadir Durrani, Fahim Dalvi, Hassan Sajjad, and James Glass. 2017. [What do Neural Machine Translation Models Learn about Morphology?](http://aclweb.org/anthology/P17-1080). In *Proceedings of ACL 2017*.
* Ella Rabinovich, Noam Ordan, and Shuly Wintner. 2017. [Found in Translation: Reconstructing Phylogenetic Language Trees from Translations](http://aclweb.org/anthology/P17-1049). In *Proceedings of ACL 2017*.
* Rico Sennrich. 2017. [How Grammatical is Character-level Neural Machine Translation? Assessing MT Quality with Contrastive Translation Pairs](http://aclweb.org/anthology/E17-2060). In *Proceedings of EACL 2017*.
* Adam Poliak, Yonatan Belinkov, James Glass, and Benjamin Van Durme. 2018. [On the Evaluation of Semantic Phenomena in Neural Machine Translation Using Natural Language Inference](http://aclweb.org/anthology/N18-2082). In *Proceedings of NAACL 2018*.
* Arianna Bisazza and Clara Tump. 2018. [The Lazy Encoder: A Fine-Grained Analysis of the Role of Morphology in Neural Machine Translation](http://aclweb.org/anthology/D18-1313). In *Proceedings of EMNLP 2018*.
* Lijun Wu, Xu Tan, Di He, Fei Tian, Tao Qin, Jianhuang Lai, and Tie-Yan Liu. 2018. [Beyond Error Propagation in Neural Machine Translation: Characteristics of Language Also Matter](http://aclweb.org/anthology/D18-1396). In *Proceedings of EMNLP 2018*.

<h3 id="fairness_and_diversity">Fairness and Diversity</h3>

* Hayahide Yamagishi, Shin Kanouchi, Takayuki Sato, and Mamoru Komachi. 2016. [Controlling the Voice of a Sentence in Japanese-to-English Neural Machine Translation](http://www.aclweb.org/anthology/W16-4620). In *Proceedings of the 3rd Workshop on Asian Translation*.          
* Rico Sennrich, Barry Haddow and Alexandra Birch. 2016. [Controlling Politeness in Neural Machine Translation via Side Constraints](http://aclweb.org/anthology/N16-1005). In *Proceedings of NAACL 2016*.  
* Xing Niu, Marianna Martindale, and Marine Carpuat. 2017. [A Study of Style in Machine Translation: Controlling the Formality of Machine Translation Output](http://aclweb.org/anthology/D17-1299). In *Proceedings of EMNLP 2016*.   
* Ella Rabinovich, Raj Nath Patel, Shachar Mirkin, Lucia Specia, and Shuly Wintner. 2017. [Personalized Machine Translation: Preserving Original Author Traits](http://aclweb.org/anthology/E17-1101). In *Proceedings of EACL 2017*.   
* Myle Ott, Michael Auli, David Grangier, and Marc'Aurelio Ranzato. 2018. [Analyzing Uncertainty in Neural Machine Translation](https://arxiv.org/pdf/1803.00047). In *Proceedings of ICML 2018*.
* Paul Michel and Graham Neubig. 2018. [Extreme Adaptation for Personalized Neural Machine Translation](http://www.aclweb.org/anthology/P18-2050). In *Proceedings of ACL 2018*.     
* Philip Schulz, Wilker Aziz, and Trevor Cohn. 2018. [A Stochastic Decoder for Neural Machine Translation](http://aclweb.org/anthology/P18-1115). In *Proceedings of ACL 2018*.
* Eva Vanmassenhove, Christian Hardmeier, and Andy Way. 2018. [Getting Gender Right in Neural Machine Translation](http://www.aclweb.org/anthology/D18-1334). In *Proceedings of EMNLP 2018*.     

<h3 id="efficiency">Efficiency</h3>

* Abigail See, Minh-Thang Luong, and Christopher D. Manning. 2016. [Compression of Neural Machine Translation Models via Pruning](http://aclweb.org/anthology/K16-1029). In *Proceedings of CoNLL 2016*.
* Yusuke Oda, Philip Arthur, Graham Neubig, Koichiro Yoshino, and Satoshi Nakamura. 2017. [Neural Machine Translation via Binary Code Prediction](http://aclweb.org/anthology/P17-1079). In *Proceedings of ACL 2017*.
* Xing Shi and Kevin Knight. 2017. [Speeding Up Neural Machine Translation Decoding by Shrinking Run-time Vocabulary](http://aclweb.org/anthology/P17-2091). In *Proceedings of ACL 2017*.
* Xiaowei Zhang, Wei Chen, Feng Wang, Shuang Xu, and Bo Xu. 2017. [Towards Compact and Fast Neural Machine Translation Using a Combined Method](http://aclweb.org/anthology/D17-1154). In *Proceedings of EMNLP 2017*. 
* Felix Stahlberg and Bill Byrne. 2017. [Unfolding and Shrinking Neural Machine Translation Ensembles](http://aclweb.org/anthology/D17-1208). In *Proceedings of EMNLP 2017*. 
* Jacob Devlin. 2017. [Sharp Models on Dull Hardware: Fast and Accurate Neural Machine Translation Decoding on the CPU](http://aclweb.org/anthology/D17-1300). In *Proceedings of EMNLP 2017*.  
* Dakun Zhang, Jungi Kim, Josep Crego, and Jean Senellart. 2017. [Boosting Neural Machine Translation](http://aclweb.org/anthology/I17-2046). In *Proceedings of IJCNLP 2017*.
* Łukasz Kaiser, Aurko Roy, Ashish Vaswani, Niki Parmar, Samy Bengio, Jakob Uszkoreit, and Noam Shazeer. 2018. [Fast Decoding in Sequence Models Using Discrete Latent Variables](https://arxiv.org/pdf/1803.03382.pdf). In *Proceedings of ICML 2018*.
* Gonzalo Iglesias, William Tambellini, Adrià de Gispert, Eva Hasler, and Bill Byrne. 2018. [Accelerating NMT Batched Beam Decoding with LMBR Posteriors for Deployment](http://aclweb.org/anthology/N18-3013). In *Proceedings of NAACL 2018*. 
* Jerry Quinn and Miguel Ballesteros. 2018. [Pieces of Eight: 8-bit Neural Machine Translation](http://aclweb.org/anthology/N18-3014). In *Proceedings of NAACL 2018*.
* Matt Post and David Vilar. 2018. [Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation](http://aclweb.org/anthology/N18-1119). In *Proceedings of NAACL 2018*.
* Biao Zhang, Deyi Xiong, and Jinsong Su. 2018. [Accelerating Neural Transformer via an Average Attention Network](http://aclweb.org/anthology/P18-1166). In *Proceedings of ACL 2018*.
* Rui Wang, Masao Utiyama, and Eiichiro Sumita. 2018. [Dynamic Sentence Sampling for Efficient Training of Neural Machine Translation](http://aclweb.org/anthology/P18-2048). In *Proceedings of ACL 2018*.
* Myle Ott, Sergey Edunov, David Grangier, and Michael Auli. 2018. [Scaling Neural Machine Translation](http://aclweb.org/anthology/W18-6301). In *Proceedings of the Third Conference on Machine Translation: Research Papers*.
* Joern Wuebker, Patrick Simianer, and John DeNero. 2018. [Compact Personalized Models for Neural Machine Translation](http://aclweb.org/anthology/D18-1104). In *Proceedings of EMNLP 2018*.
* Wen Zhang, Liang Huang, Yang Feng, Lei Shen, and Qun Liu. 2018. [Speeding Up Neural Machine Translation Decoding by Cube Pruning](http://aclweb.org/anthology/D18-1460). In *Proceedings of EMNLP 2018*.  
* Zhisong Zhang, Rui Wang, Masao Utiyama, Eiichiro Sumita, and Hai Zhao. 2018. [Exploring Recombination for Efficient Decoding of Neural Machine Translation](http://aclweb.org/anthology/D18-1511). In *Proceedings of EMNLP 2018*.   
* Nikolay Bogoychev, Kenneth Heafield, Alham Fikri Aji, and Marcin Junczys-Dowmunt. 2018. [Accelerating Asynchronous Stochastic Gradient Descent for Neural Machine Translation](http://aclweb.org/anthology/D18-1332). In *Proceedings of EMNLP 2018*.   
* Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. 2018. [Blockwise Parallel Decoding for Deep Autoregressive Models](https://papers.nips.cc/paper/8212-blockwise-parallel-decoding-for-deep-autoregressive-models.pdf). In *Proceedings of NeurIPS 2018*.


<h3 id="pre_training">Pre-Training</h3>

* Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 2017. [Learned in Translation: Contextualized Word Vectors](http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf). In *Proceedings of NIPS 2017*.
* Ye Qi, Devendra Sachan, Matthieu Felix, Sarguna Padmanabhan, and Graham Neubig. 2018. [When and Why Are Pre-Trained Word Embeddings Useful for Neural Machine Translation?](http://aclweb.org/anthology/N18-2084). In *Proceedings of NAACL 2018*.
* Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. [Deep Contextualized Word Representations](http://aclweb.org/anthology/N18-1202). In *Proceedings of NAACL 2018*.
* Jeremy Howard and Sebastian Ruder. 2018. [Universal Language Model Fine-tuning for Text Classification](http://aclweb.org/anthology/P18-1031). In *Proceedings of ACL 2018*.
* Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). Technical Report, OpenAI.
* Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805). *arXiv:1810.04805*.

<h3 id="speech_translation_and_simultaneous_translation">Speech Translation and Simultaneous Translation</h3>

* Matt Post, Gaurav Kumar, Adam Lopez, Damianos Karakos, Chris Callison-Burch and Sanjeev Khudanpur. 2013. [Improved Speech-to-Text Translation with the Fisher and Callhome Spanish–English Speech Translation Corpus](http://www.mt-archive.info/10/IWSLT-2013-Post.pdf). In *Proceedings of IWSLT 2013*.  
* Gaurav Kumar, Matt Post, Daniel Povey and Sanjeev Khudanpur. 2014. [Some insights from translating conversational telephone speech](https://ieeexplore.ieee.org/abstract/document/6854197) In *Proceedings of ICASSP 2014*. 
* Long Duong, Antonios Anastasopoulos, David Chiang, Steven Bird, and Trevor Cohn. 2016. [An Attentional Model for Speech Translation without Transcription](http://www.aclweb.org/anthology/N16-1109). In *Proceedings of NAACL 2016*.
* Antonios Anastasopoulos, David Chiang, and Long Duong. 2016. [An Unsupervised Probability Model for Speech-to-translation Alignment of Low-resource Languages](https://aclweb.org/anthology/D16-1133). In *Proceedings of EMNLP 2016*.
* Ron J. Weiss, Jan Chorowski, Navdeep Jaitly, Yonghui Wu and Zhifeng Chen. 2017. [Sequence-to-sequence models can directly translate foreign speech](https://arxiv.org/abs/1703.08581). In *Proceedings of Interspeech 2017*.
* Jiatao Gu, Graham Neubig, Kyunghyun Cho, and Victor O.K. Li. 2017. [Learning to Translate in Real-time with Neural Machine Translation](http://aclweb.org/anthology/E17-1099). In *Proceedings of EACL 2017*.
* Sameer Bansal, Herman Kamper, Adam Lopez, and Sharon Goldwater. 2017. [Towards speech-to-text translation without speech recognition](http://aclweb.org/anthology/E17-2076). In *Proceedings of EACL 2017*.
* Jiatao Gu, James Bradbury, Caiming Xiong, Victor O.K. Li, and Richard Socher. 2018. [Non-Autoregressive Neural Machine Translation](https://openreview.net/pdf?id=B1l8BtlCb). In *Proceedings of ICLR 2018*. 
* Antonios Anastasopoulos and David Chiang. 2018. [Tied Multitask Learning for Neural Speech Translation](https://arxiv.org/pdf/1802.06655.pdf). In *Proceedings of NAACL 2018*.
* Fahim Dalvi, Nadir Durrani, Hassan Sajjad, and Stephan Vogel. 2018. [Incremental Decoding and Training Methods for Simultaneous Translation in Neural Machine Translation](http://aclweb.org/anthology/N18-2079). In *Proceedings of NAACL 2018*.   
* Craig Stewart, Nikolai Vogler, Junjie Hu, Jordan Boyd-Graber, and Graham Neubig. 2018. [Automatic Estimation of Simultaneous Interpreter Performance](http://aclweb.org/anthology/P18-2105). In *Proceedings of ACL 2018*.
* Florian Dessloch, Thanh-Le Ha, Markus Müller, Jan Niehues, Thai Son Nguyen, Ngoc-Quan Pham, Elizabeth Salesky, Matthias Sperber, Sebastian Stüker, Thomas Zenkel, and Alexander Waibel. 2018. [KIT Lecture Translator: Multilingual Speech Translation with One-Shot Learning](http://aclweb.org/anthology/C18-2020). In *Proceedings of COLING 2018*.
* Chunqi Wang, Ji Zhang, and Haiqing Chen. 2018. [Semi-Autoregressive Neural Machine Translation](http://aclweb.org/anthology/D18-1044). In *Proceedings of EMNLP 2018*.  
* Jindřich Libovický and Jindřich Helcl. 2018. [End-to-End Non-Autoregressive Neural Machine Translation with Connectionist Temporal Classification](http://aclweb.org/anthology/D18-1336). In *Proceedings of EMNLP 2018*. 
* Ashkan Alinejad, Maryam Siahbani, and Anoop Sarkar. 2018. [Prediction Improves Simultaneous Neural Machine Translation](http://aclweb.org/anthology/D18-1337). In *Proceedings of EMNLP 2018*. 
* Mingbo Ma, Liang Huang, Hao Xiong, Kaibo Liu, Chuanqiang Zhang, Zhongjun He, Hairong Liu, Xing Li, and Haifeng Wang. 2018. [STACL: Simultaneous Translation with Integrated Anticipation and Controllable Latency](https://arxiv.org/pdf/1810.08398). *arXiv:1810.08398*.

<h3 id="multi_modality">Multi-modality</h3>

* Lucia Specia, Stella Frank, Khalil Sima'an, and Desmond Elliott. 2016. [A Shared Task on Multimodal Machine Translation and Crosslingual Image Description](http://aclweb.org/anthology/W16-2346). In *Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers*.
* Sergio Rodríguez Guasch, Marta R. Costa-jussà. 2016. [WMT 2016 Multimodal Translation System Description based on Bidirectional Recurrent Neural Networks with Double-Embeddings](http://aclweb.org/anthology/W16-2362). In *Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers*.
* Po-Yao Huang, Frederick Liu, Sz-Rung Shiang, Jean Oh, and Chris Dyer. 2016. [Attention-based Multimodal Neural Machine Translation](http://aclweb.org/anthology/W16-2363). In *Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers*.
* Iacer Calixto, Desmond Elliott, and Stella Frank. 2016. [DCU-UvA Multimodal MT System Report](http://aclweb.org/anthology/W16-2359). In *Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers*.
* Desmond Elliott, Stella Frank, Loïc Barrault, Fethi Bougares, and Lucia Specia. 2017. [Findings of the Second Shared Task on Multimodal Machine Translation and Multilingual Image Description](http://aclweb.org/anthology/W17-4718). In *Proceedings of the Second Conference on Machine Translation*.
* Iacer Calixto, Qun Liu, and Nick Campbell. 2017. [Doubly-Attentive Decoder for Multi-modal Neural Machine Translation](http://aclweb.org/anthology/P17-1175). In *Proceedings of ACL 2017*.
* Jean-Benoit Delbrouck and Stéphane Dupont. 2017. [An empirical study on the effectiveness of images in Multimodal Neural Machine Translation](http://aclweb.org/anthology/D17-1095). In *Proceedings of EMNLP 2017*.
* Iacer Calixto and Qun Liu. 2017. [Incorporating Global Visual Features into Attention-based Neural Machine Translation](http://aclweb.org/anthology/D17-1105). In *Proceedings of EMNLP 2017*. 
* Jason Lee, Kyunghyun Cho, Jason Weston, and Douwe Kiela. 2018. [Emergent Translation in Multi-Agent Communication](https://openreview.net/pdf?id=H1vEXaxA-). In *Proceedings of ICLR 2018*.
* Yun Chen, Yang Liu, and Victor O. K. Li. 2018. [Zero-Resource Neural Machine Translation with Multi-Agent Communication Game](https://arxiv.org/pdf/1802.03116). In *Proceedings of AAAI 2018*.
* Loïc Barrault, Fethi Bougares, Lucia Specia, Chiraag Lala, Desmond Elliott, and Stella Frank. 2018. [Findings of the Third Shared Task on Multimodal Machine Translation](http://aclweb.org/anthology/W18-6402). In *Proceedings of the Third Conference on Machine Translation: Shared Task Papers*.
* John Hewitt, Daphne Ippolito, Brendan Callahan, Reno Kriz, Derry Tanti Wijaya, and Chris Callison-Burch. 2018. [Learning Translations via Images with a Massively Multilingual Image Dataset](http://aclweb.org/anthology/P18-1239). In *Proceedings of ACL 2018*.
* Mingyang Zhou, Runxiang Cheng, Yong Jae Lee, and Zhou Yu. 2018. [A Visual Attention Grounding Neural Model for Multimodal Machine Translation](http://aclweb.org/anthology/D18-1400). In *Proceedings of EMNLP 2018*.
* Desmond Elliott. 2018. [Adversarial Evaluation of Multimodal Machine Translation](http://aclweb.org/anthology/D18-1329). In *Proceedings of EMNLP 2018*.

<h3 id="domain_adaptation">Domain Adaptation</h3>

* Chenhui Chu, Raj Dabre, and Sadao Kurohashi. 2017. [An Empirical Comparison of Domain Adaptation Methods for Neural Machine Translation](http://aclweb.org/anthology/P17-2061). In *Proceedings of ACL 2017*.
* Rui Wang, Andrew Finch, Masao Utiyama, and Eiichiro Sumita. 2017. [Sentence Embedding for Neural Machine Translation Domain Adaptation](http://aclweb.org/anthology/P17-2089). In *Proceedings of ACL 2017*.
* Boxing Chen, Colin Cherry, George Foster, and Samuel Larkin. 2017. [Cost Weighting for Neural Machine Translation Domain Adaptation](http://aclweb.org/anthology/W17-3205). In *Proceedings of the First Workshop on Neural Machine Translation*.
* Reid Pryzant and Denny Britz. 2017. [Effective Domain Mixing for Neural Machine Translation](http://aclweb.org/anthology/W17-4712). In *Proceedings of the Second Conference on Machine Translation*.
* Rui Wang, Masao Utiyama, Lemao Liu, Kehai Chen, and Eiichiro Sumita. 2017. [Instance Weighting for Neural Machine Translation Domain Adaptation](http://aclweb.org/anthology/D17-1155). In *Proceedings of EMNLP 2017*.
* Antonio Valerio Miceli Barone, Barry Haddow, Ulrich Germann, and Rico Sennrich. 2017. [Regularization techniques for fine-tuning in neural machine translation](http://aclweb.org/anthology/D17-1156). In *Proceedings of EMNLP 2017*.
* David Vilar. 2018. [Learning Hidden Unit Contribution for Adapting Neural Machine Translation Models](http://aclweb.org/anthology/N18-2080). In *Proceedings of NAACL 2018*.    
* Shiqi Zhang and Deyi Xiong. 2018. [Sentence Weighting for Neural Machine Translation Domain Adaptation](http://aclweb.org/anthology/C18-1269). In *Proceedings of COLING 2018*.    
* Chenhui Chu and Rui Wang. 2018. [A Survey of Domain Adaptation for Neural Machine Translation](http://aclweb.org/anthology/C18-1111). In *Proceedings of COLING 2018*.
* Jiali Zeng, Jinsong Su, Huating Wen, Yang Liu, Jun Xie, Yongjing Yin, and Jianqiang Zhao. 2018. [Multi-Domain Neural Machine Translation with Word-Level Domain Context Discrimination](http://aclweb.org/anthology/D18-1041). In *Proceedings of EMNLP 2018*.
* Graham Neubig and Junjie Hu. 2018. [Rapid Adaptation of Neural Machine Translation to New Languages](http://aclweb.org/anthology/D18-1103). In *Proceedings of EMNLP 2018*.  

<h3 id="quality_estimation">Quality Estimation</h3>

* Hyun Kim and Jong-Hyeok Lee. 2016. [A Recurrent Neural Networks Approach for Estimating the Quality of Machine Translation Output](http://aclweb.org/anthology/N16-1059). In *Proceedings of NAACL 2016*.
* Hyun Kim and Jong-Hyeok Lee, Seung-Hoon Na. 2017. [Predictor-Estimator using Multilevel Task Learning with Stack Propagation for Neural Quality Estimation](http://aclweb.org/anthology/W17-4763). In *Proceedings of WMT 2017*.
* Osman Baskaya, Eray Yildiz, Doruk Tunaoglu, Mustafa Tolga Eren, and A. Seza Doğruöz. 2017. [Integrating Meaning into Quality Evaluation of Machine Translation](http://aclweb.org/anthology/E17-1020). In *Proceedings of EACL 2017*. 
* Yvette Graham, Qingsong Ma, Timothy Baldwin, Qun Liu, Carla Parra, and Carolina Scarton. 2017. [Improving Evaluation of Document-level Machine Translation Quality Estimation](http://aclweb.org/anthology/E17-2057). In *Proceedings of EACL 2017*.    
* Rico Sennrich. 2017. [How Grammatical is Character-level Neural Machine Translation? Assessing MT Quality with Contrastive Translation Pairs](http://aclweb.org/anthology/E17-2060). In *Proceedings of EACL 2017*.
* Pierre Isabelle, Colin Cherry, and George Foster. 2017. [A Challenge Set Approach to Evaluating Machine Translation](http://aclweb.org/anthology/D17-1263). In *Proceedings of EMNLP 2017*. 
* André F.T. Martins, Marcin Junczys-Dowmunt, Fabio N. Kepler, Ramón Astudillo, Chris Hokamp, and Roman Grundkiewicz. 2017. [Pushing the Limits of Translation Quality Estimation](http://aclweb.org/anthology/Q17-1015). *Transactions of the Association for Computational Linguistics*.
* Maoxi Li, Qingyu Xiang, Zhiming Chen, and Mingwen Wang. 2018. [A Unified Neural Network for Quality Estimation of Machine Translation](https://www.jstage.jst.go.jp/article/transinf/E101.D/9/E101.D_2018EDL8019/_article/-char/en). *IEICE Transactions on Information and Systems*.
* Lucia Specia, Frédéric Blain, Varvara Logacheva, Ramón F. Astudillo, and André Martins. 2018. [Findings of the WMT 2018 Shared Task on Quality Estimation](http://aclweb.org/anthology/W18-6451). In *Proceedings of WMT 2018*.
* Craig Stewart, Nikolai Vogler, Junjie Hu, Jordan Boyd-Graber, and Graham Neubig. 2018. [Automatic Estimation of Simultaneous Interpreter Performance](http://aclweb.org/anthology/P18-2105). In *Proceedings of ACL 2018*.
* Julia Ive, Frédéric Blain, and Lucia Specia. 2018. [deepQuest: A Framework for Neural-based Quality Estimation](http://aclweb.org/anthology/C18-1266). In *Proceedings of COLING 2018*. 
* Kai Fan, Jiayi Wang, Bo Li, Fengming Zhou, Boxing Chen, and Luo Si. 2019. ["Bilingual Expert" Can Find Translation Errors](https://arxiv.org/pdf/1807.09433). In *Proceedings of AAAI 2019*.

<h3 id="ape">Automatic Post-Editing</h3>

* Santanu Pal, Sudip Kumar Naskar, Mihaela Vela, and Josef van Genabith. 2016. [A neural network based approach to automatic post-editing](http://aclweb.org/anthology/P16-2046). In *Proceedings of ACL 2016*. 
* Marcin Junczys-Dowmunt and Roman Grundkiewicz. 2016. [Log-linear Combinations of Monolingual and Bilingual Neural Machine Translation Models for Automatic Post-Editing](http://aclweb.org/anthology/W16-2378). In *Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers*.
* Santanu Pal, Sudip Kumar Naskar, Mihaela Vela, Qun Liu, and Josef van Genabith. 2017. [Neural Automatic Post-Editing Using Prior Alignment and Reranking](http://aclweb.org/anthology/E17-2056). In *Proceedings of EACL 2017*.
* Rajen Chatterjee, Gebremedhen Gebremelak, Matteo Negri, and Marco Turchi. 2017. [Online Automatic Post-editing for MT in a Multi-Domain Translation Environment](http://aclweb.org/anthology/E17-1050). In *Proceedings of EACL 2017*.
* David Grangier and Michael Auli. 2018. [QuickEdit: Editing Text & Translations by Crossing Words Out](http://aclweb.org/anthology/N18-1025).  In *Proceedings of NAACL 2018*.
* Thuy-Trang Vu and Gholamreza Haffari. 2018. [Automatic Post-Editing of Machine Translation: A Neural Programmer-Interpreter Approach](http://aclweb.org/anthology/D18-1341). In *Proceedings of EMNLP 2018*.

<h3 id="word_translation">Word Translation and Bilingual Lexicon Induction</h3>

* Tomas Mikolov, Quoc V. Le, and Ilya Sutskever. 2013. [Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/pdf/1309.4168.pdf). *arxiv:1309.4168*.
* Georgiana Dinu, Angeliki Lazaridou, and Marco Baroni. 2015. [Improving Zero-shot Learning by Mitigating the Hubness Problem](https://arxiv.org/pdf/1412.6568.pdf). In *Proceedings of ICLR 2015*.
* Meng Zhang, Yang Liu, Huanbo Luan, Maosong Sun, Tatsuya Izuha, and Jie Hao. 2016. [Building Earth Mover's Distance on Bilingual Word Embeddings for Machine Translation](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12227/12035). In *Proceedings of AAAI 2016*.
* Meng Zhang, Yang Liu, Huanbo Luan, Yiqun Liu, and Maosong Sun. 2016. [Inducing Bilingual Lexica From Non-Parallel Data With Earth Mover's Distance Regularization](http://aclweb.org/anthology/C16-1300). In *Proceedings of COLING 2016*.
* Ivan Vulić and Anna Korhonen. [On the Role of Seed Lexicons in Learning Bilingual Word Embeddings](http://www.aclweb.org/anthology/P16-1024). In *Proceedings of ACL 2016*. 
* Meng Zhang, Haoruo Peng, Yang Liu, Huanbo Luan, and Maosong Sun. [Bilingual Lexicon Induction from Non-Parallel Data with Minimal Supervision](http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14682/14264). In *Proceedings of AAAI 2017*.
* Ann Irvine and Chris Callison-Burch. 2017. [A Comprehensive Analysis of Bilingual Lexicon Induction](http://aclweb.org/anthology/J17-2001). *Computational Linguistics*.
* Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2017. [Learning bilingual word embeddings with (almost) no bilingual data](http://aclweb.org/anthology/P17-1042). In *Proceedings of ACL 2017*.
* Meng Zhang, Yang Liu, Huanbo Luan, and Maosong Sun. 2017. [Adversarial Training for Unsupervised Bilingual Lexicon Induction](http://aclweb.org/anthology/P17-1179). In *Proceedings of ACL 2017*.
* Geert Heyman, Ivan Vulić, and Marie-Francine Moens. 2017. [Bilingual Lexicon Induction by Learning to Combine Word-Level and Character-Level Representations](http://aclweb.org/anthology/E17-1102). In *Proceedings of EACL 2017*.
* Bradley Hauer, Garrett Nicolai, and Grzegorz Kondrak. 2017. [Bootstrapping Unsupervised Bilingual Lexicon Induction](http://aclweb.org/anthology/E17-2098). In *Proceedings of EACL 2017*. 
* Yunsu Kim, Julian Schamper, and Hermann Ney. 2017. [Unsupervised Training for Large Vocabulary Translation Using Sparse Lexicon and Word Classes](http://aclweb.org/anthology/E17-2103). In *Proceedings of EACL 2017*.  
* Derry Tanti Wijaya, Brendan Callahan, John Hewitt, Jie Gao, Xiao Ling, Marianna Apidianaki, and Chris Callison-Burch. 2017. [Learning Translations via Matrix Completion](http://aclweb.org/anthology/D17-1152). In *Proceedings of EMNLP 2017*.
* Meng Zhang, Yang Liu, Huanbo Luan, and Maosong Sun. 2017. [Earth Mover's Distance Minimization for Unsupervised Bilingual Lexicon Induction](http://aclweb.org/anthology/D17-1207). In *Proceedings of EMNLP 2017*.
* Ndapandula Nakashole and Raphael Flauger. 2017. [Knowledge Distillation for Bilingual Dictionary Induction](http://aclweb.org/anthology/D17-1264). In *Proceedings of EMNLP 2017*. 
* Guillaume Lample, Alexis Conneau, Marc'Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2018. [Word translation without parallel data](https://openreview.net/pdf?id=H196sainb). In *Proceedings of ICLR 2018*.
* Fabienne Braune, Viktor Hangya, Tobias Eder, and Alexander Fraser. 2018. [Evaluating bilingual word embeddings on the long tail](http://aclweb.org/anthology/N18-2030). In *Proceedings of NAACL 2018*. 
* Ndapa Nakashole and Raphael Flauger. 2018. [Characterizing Departures from Linearity in Word Translation](http://aclweb.org/anthology/P18-2036). In *Proceedings of ACL 2018*. 
* Anders Søgaard, Sebastian Ruder, and Ivan Vulić. 2018. [On the Limitations of Unsupervised Bilingual Dictionary Induction](http://aclweb.org/anthology/P18-1072). In *Proceedings of ACL 2018*.
* Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. [A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings](http://aclweb.org/anthology/P18-1073). In *Proceedings of ACL 2018*. 
* Parker Riley and Daniel Gildea. 2018. [Orthographic Features for Bilingual Lexicon Induction](http://aclweb.org/anthology/P18-2062). In *Proceedings of ACL 2018*.
* Amir Hazem and Emmanuel Morin. 2018. [Leveraging Meta-Embeddings for Bilingual Lexicon Extraction from Specialized Comparable Corpora](http://aclweb.org/anthology/C18-1080). In *Proceedings of COLING 2018*.
* Xilun Chen and Claire Cardie. 2018. [Unsupervised Multilingual Word Embeddings](http://aclweb.org/anthology/D18-1024). In *Proceedings of EMNLP 2018*.
* Yerai Doval, Jose Camacho-Collados, Luis Espinosa Anke, and Steven Schockaert. 2018. [Improving Cross-Lingual Word Embeddings by Meeting in the Middle](http://aclweb.org/anthology/D18-1027). In *Proceedings of EMNLP 2018*. 
* Sebastian Ruder, Ryan Cotterell, Yova Kementchedjhieva, and Anders Søgaard. 2018. [A Discriminative Latent-Variable Model for Bilingual Lexicon Induction](http://aclweb.org/anthology/D18-1042). In *Proceedings of EMNLP 2018*.
* Ndapa Nakashole. 2018. [NORMA: Neighborhood Sensitive Maps for Multilingual Word Embeddings](http://aclweb.org/anthology/D18-1047). In *Proceedings of EMNLP 2018*.
* Zi-Yi Dou, Zhi-Hao Zhou, and Shujian Huang. 2018. [Unsupervised Bilingual Lexicon Induction via Latent Variable Models](http://aclweb.org/anthology/D18-1062). In *Proceedings of EMNLP 2018*.
* Armand Joulin, Piotr Bojanowski, Tomas Mikolov, Hervé Jégou, and Edouard Grave. 2018. [Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion](http://aclweb.org/anthology/D18-1330). In *Proceedings of EMNLP 2018*.
* Sebastian Ruder, Ivan Vulić, and Anders Søgaard. 2018. [A Survey Of Cross-lingual Word Embedding Models](https://arxiv.org/pdf/1706.04902.pdf). *arxiv:1706.04902*.

<h3 id="poetry_translation">Poetry Translation</h3>

* Marjan Ghazvininejad, Yejin Choi, and Kevin Knight. 2018. [Neural Poetry Translation](http://aclweb.org/anthology/N18-2011). In *Proceedings of NAACL 2018*.
