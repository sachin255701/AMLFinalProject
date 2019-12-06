AML Final Project

Team Members:
1. Richa Singh (University ID: 906252623)
2. FNU Sachin (University ID: 906270534)

__NOTE: We are not able to upload emmist-balanced.mat file which is EMNIST's dataset due to large file size. This dataset can be found on link http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip. If not able to find please send us an email on richas@vt.edu or sachin255701@vt.edu.

Contents of Project:
1. src/main.ipynb: Pyhton notebook to run the code mentioned in PDF report.
2. src/emmist-balanced.mat: MATLAB format of EMNIST dataset (not able to upload due to size more than 25 MB).
3. src/adversarial_examples.py: Script to generate various adversarial examples.
4. src/plots.py: Script to plot various analysis plots.
5. src/train_network.py: Script to train basic and Distilled neural networks.
6. src/l0_attack.py: Author's code modified for our infrastructure. Contains L-0 attack.
7. src/l2_attack.py: Author's code modified for our infrastructure. Contains L-2 attack.
8. src/li_attack.py: Author's code modified for our infrastructure. Contains L-Infinity attack.
9. src/models: Directory that contains various trained Defensive Distillation neurall network models.
10. ProjectReport.pdf: Project report which contains all the details about project.
  

Topics targeted in Project:
1. Generation and Analysis of Adversarial Examples.
2. Implementation of Defensive Distillation Deep Neural Networks and analysis of its behavior with Adversarial Examples.
3. Verification of three attack models mentioned in [5].


Summary of Project: We have proved several claims from the papers [1], [3] and [5] using EMNIST data set as given below:
1. We have observed that as the value of ϵ increases, the adversarial sample success rate also increases which means a greater number of input images are getting classified into incorrect output label. Here, adversarial samples are generated using FGSM technique implemented by us. This point is proven by paper [1]. 
2. Next, we have observed that as the distillation temperature increases, the adversarial sample success rate decreases which means that distilled network implemented by us is able to provide resilience towards adversarial examples and model’s sensitivity to adversarial perturbations decreases with increase in distillation temperature. This point is proven by paper [3]. 
3. We have also observed that the neural network trained at different distillation temperatures has only moderate effect on the classification accuracy of the un-distilled model. This is another point proven by paper [3].
4. Finally, we have observed that  L_0, L_2 and L_∞ attacks generate such high confidence adversarial examples that they are able to fail the defensive distillation neural network trained at different temperatures which was otherwise providing security against previous attacks but not against these three attacks. So, this clearly demonstrates the fact given in paper [5] that increasing the distillation temperature does not increase the robustness of the underlying neural network.


Papers Referenced:
1. Xiaoyong Yuan, Pan He, Qile Zhu, Xiaolin Li, “Adversarial Examples: Attacks and Defenses for Deep Learning”.
2. Ian J. Goodfellow, Jonathon Shlens & Christian Szegedy, "EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES", ICLR 2015.
3. Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, and Ananthram Swami, "Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks,"37th IEEE Symposium on Security & Privacy, IEEE 2016, San Jose, CA.
4. Geoffrey Hinton, Oriol Vinyals, Jeff Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531v1 [stat.ML] 9 Mar 2015.
5. Nicholas Carlini, David Wagner, "Towards Evaluating the Robustness of Neural Networks," arXiv:1608.04644v2 [cs.CR] 22 Mar 2017.
